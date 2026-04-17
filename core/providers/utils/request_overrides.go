package utils

import (
	"fmt"

	"github.com/google/cel-go/cel"
	schemas "github.com/maximhq/bifrost/core/schemas"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

type compiledOverride struct {
	rule    schemas.RequestOverride
	program cel.Program
}

// RequestOverrideEngine holds pre-compiled CEL programs for request parameter overrides.
type RequestOverrideEngine struct {
	overrides []compiledOverride
}

// NewRequestOverrideEngine compiles all CEL expressions and returns a ready-to-use engine.
func NewRequestOverrideEngine(overrides []schemas.RequestOverride) (*RequestOverrideEngine, error) {
	if len(overrides) == 0 {
		return nil, nil
	}

	env, err := cel.NewEnv(
		cel.Variable("model", cel.StringType),
		cel.Variable("request_type", cel.StringType),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create CEL environment: %w", err)
	}

	compiled := make([]compiledOverride, 0, len(overrides))
	for i, o := range overrides {
		var prog cel.Program
		if o.Match == "" {
			// Empty match = always true
			ast, iss := env.Compile("true")
			if iss.Err() != nil {
				return nil, fmt.Errorf("request_overrides[%d]: failed to compile default expression: %w", i, iss.Err())
			}
			prog, err = env.Program(ast)
			if err != nil {
				return nil, fmt.Errorf("request_overrides[%d]: failed to create program: %w", i, err)
			}
		} else {
			ast, iss := env.Compile(o.Match)
			if iss.Err() != nil {
				return nil, fmt.Errorf("request_overrides[%d]: invalid CEL expression %q: %w", i, o.Match, iss.Err())
			}
			prog, err = env.Program(ast)
			if err != nil {
				return nil, fmt.Errorf("request_overrides[%d]: failed to create program for %q: %w", i, o.Match, err)
			}
		}
		compiled = append(compiled, compiledOverride{rule: o, program: prog})
	}

	return &RequestOverrideEngine{overrides: compiled}, nil
}

// ApplyOverrides evaluates all rules against the given model/requestType and applies
// matching overrides to the JSON body. Rules are applied in order; all matching rules run.
func (e *RequestOverrideEngine) ApplyOverrides(jsonBody []byte, model string, requestType string) ([]byte, error) {
	if e == nil || len(e.overrides) == 0 {
		return jsonBody, nil
	}

	vars := map[string]interface{}{
		"model":        model,
		"request_type": requestType,
	}

	for _, co := range e.overrides {
		matched, err := evaluateOverrideCEL(co.program, vars)
		if err != nil {
			return nil, fmt.Errorf("evaluating CEL expression %q: %w", co.rule.Match, err)
		}
		if !matched {
			continue
		}

		// Apply operations in order: defaults → set → remove
		if len(co.rule.Defaults) > 0 {
			jsonBody, err = applyDefaults(jsonBody, co.rule.Defaults)
			if err != nil {
				return nil, fmt.Errorf("applying defaults: %w", err)
			}
		}
		if len(co.rule.Set) > 0 {
			jsonBody, err = applySet(jsonBody, co.rule.Set)
			if err != nil {
				return nil, fmt.Errorf("applying set: %w", err)
			}
		}
		if len(co.rule.Remove) > 0 {
			jsonBody, err = applyRemove(jsonBody, co.rule.Remove)
			if err != nil {
				return nil, fmt.Errorf("applying remove: %w", err)
			}
		}
	}

	return jsonBody, nil
}

// ApplyRequestOverridesFromContext is a convenience function that extracts the override engine
// from context, infers the request type, and applies overrides to the JSON body.
// This eliminates code duplication across provider implementations.
func ApplyRequestOverridesFromContext(ctx *schemas.BifrostContext, jsonBody []byte, inferredRequestType schemas.RequestType) ([]byte, error) {
	engine, _ := ctx.Value(schemas.BifrostContextKeyRequestOverrideEngine).(*RequestOverrideEngine)
	if engine == nil {
		return jsonBody, nil
	}

	// Try to get request type from context first (set by bifrost-http transport)
	requestType, _ := ctx.Value(schemas.BifrostContextKeyHTTPRequestType).(schemas.RequestType)
	// If not in context, use the inferred type from caller
	if requestType == "" {
		requestType = inferredRequestType
	}

	model := ExtractModelFromJSON(jsonBody)
	return engine.ApplyOverrides(jsonBody, model, string(requestType))
}

func evaluateOverrideCEL(program cel.Program, vars map[string]interface{}) (bool, error) {
	out, _, err := program.Eval(vars)
	if err != nil {
		return false, err
	}
	result, ok := out.Value().(bool)
	if !ok {
		return false, fmt.Errorf("CEL expression did not return a boolean")
	}
	return result, nil
}

// applyDefaults merges params into jsonBody only for keys that don't already exist.
func applyDefaults(jsonBody []byte, defaults map[string]interface{}) ([]byte, error) {
	filtered := make(map[string]interface{}, len(defaults))
	for k, v := range defaults {
		if !gjson.GetBytes(jsonBody, k).Exists() {
			filtered[k] = v
		}
	}
	if len(filtered) == 0 {
		return jsonBody, nil
	}
	return MergeExtraParamsIntoJSON(jsonBody, filtered)
}

// applySet force-merges params into jsonBody, overwriting existing values.
func applySet(jsonBody []byte, params map[string]interface{}) ([]byte, error) {
	return MergeExtraParamsIntoJSON(jsonBody, params)
}

// applyRemove deletes the specified top-level keys from the JSON body.
func applyRemove(jsonBody []byte, keys []string) ([]byte, error) {
	result := jsonBody
	for _, key := range keys {
		if gjson.GetBytes(result, key).Exists() {
			var err error
			result, err = sjson.DeleteBytes(result, key)
			if err != nil {
				return nil, fmt.Errorf("failed to remove key %q: %w", key, err)
			}
		}
	}
	return result, nil
}

// ExtractModelFromJSON extracts the "model" field from a JSON body.
func ExtractModelFromJSON(jsonBody []byte) string {
	result := gjson.GetBytes(jsonBody, "model")
	return result.String()
}
