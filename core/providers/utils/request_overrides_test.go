package utils

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
)

// --- NewRequestOverrideEngine tests ---

func TestNewRequestOverrideEngine_NilForEmpty(t *testing.T) {
	engine, err := NewRequestOverrideEngine(nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if engine != nil {
		t.Fatal("expected nil engine for empty overrides")
	}
}

func TestNewRequestOverrideEngine_CompileSuccess(t *testing.T) {
	overrides := []schemas.RequestOverride{
		{
			Match: "model.contains('claude')",
			Set:   map[string]interface{}{"group": "test"},
		},
		{
			Match:    "",
			Defaults: map[string]interface{}{"temperature": 0.7},
		},
	}
	engine, err := NewRequestOverrideEngine(overrides)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if engine == nil {
		t.Fatal("expected non-nil engine")
	}
	if len(engine.overrides) != 2 {
		t.Fatalf("expected 2 compiled overrides, got %d", len(engine.overrides))
	}
}

func TestNewRequestOverrideEngine_InvalidCEL(t *testing.T) {
	overrides := []schemas.RequestOverride{
		{Match: "this is not valid CEL !!!"},
	}
	_, err := NewRequestOverrideEngine(overrides)
	if err == nil {
		t.Fatal("expected error for invalid CEL expression")
	}
	if !strings.Contains(err.Error(), "request_overrides[0]") {
		t.Fatalf("error should reference index: %v", err)
	}
}

// --- CEL matching tests ---

func TestApplyOverrides_MatchContains(t *testing.T) {
	engine := mustBuildEngine(t, []schemas.RequestOverride{
		{
			Match: "model.contains('claude')",
			Set:   map[string]interface{}{"group": "claude-guan"},
		},
	})

	input := toJSON(t, map[string]interface{}{
		"model":   "claude-3-sonnet",
		"message": "hello",
	})

	result, err := engine.ApplyOverrides(input, "claude-3-sonnet", "chat_completion")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	parsed := parseJSON(t, result)
	if parsed["group"] != "claude-guan" {
		t.Fatalf("expected group=claude-guan, got %v", parsed["group"])
	}
}

func TestApplyOverrides_NoMatch(t *testing.T) {
	engine := mustBuildEngine(t, []schemas.RequestOverride{
		{
			Match: "model.contains('claude')",
			Set:   map[string]interface{}{"group": "claude-guan"},
		},
	})

	input := toJSON(t, map[string]interface{}{
		"model":   "gpt-4o",
		"message": "hello",
	})

	result, err := engine.ApplyOverrides(input, "gpt-4o", "chat_completion")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	parsed := parseJSON(t, result)
	if _, exists := parsed["group"]; exists {
		t.Fatal("group should not exist when model doesn't match")
	}
}

func TestApplyOverrides_EmptyMatchAlwaysApplies(t *testing.T) {
	engine := mustBuildEngine(t, []schemas.RequestOverride{
		{
			Match: "",
			Set:   map[string]interface{}{"always": true},
		},
	})

	input := toJSON(t, map[string]interface{}{"model": "anything"})
	result, err := engine.ApplyOverrides(input, "anything", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	parsed := parseJSON(t, result)
	if parsed["always"] != true {
		t.Fatalf("expected always=true, got %v", parsed["always"])
	}
}

func TestApplyOverrides_RequestTypeMatch(t *testing.T) {
	engine := mustBuildEngine(t, []schemas.RequestOverride{
		{
			Match: "request_type == 'embedding'",
			Set:   map[string]interface{}{"embed_flag": true},
		},
	})

	// Should match
	input := toJSON(t, map[string]interface{}{"model": "text-embedding-ada-002"})
	result, err := engine.ApplyOverrides(input, "text-embedding-ada-002", "embedding")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	parsed := parseJSON(t, result)
	if parsed["embed_flag"] != true {
		t.Fatal("expected embed_flag=true for embedding request type")
	}

	// Should not match
	result2, err := engine.ApplyOverrides(input, "text-embedding-ada-002", "chat_completion")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	parsed2 := parseJSON(t, result2)
	if _, exists := parsed2["embed_flag"]; exists {
		t.Fatal("embed_flag should not be set for chat_completion")
	}
}

// --- Set operation tests ---

func TestApplyOverrides_SetOverwritesExisting(t *testing.T) {
	engine := mustBuildEngine(t, []schemas.RequestOverride{
		{
			Match: "",
			Set:   map[string]interface{}{"temperature": 0.9},
		},
	})

	input := toJSON(t, map[string]interface{}{
		"model":       "gpt-4o",
		"temperature": 0.5,
	})

	result, err := engine.ApplyOverrides(input, "gpt-4o", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	parsed := parseJSON(t, result)
	if parsed["temperature"] != 0.9 {
		t.Fatalf("expected temperature=0.9, got %v", parsed["temperature"])
	}
}

func TestApplyOverrides_SetAddsNew(t *testing.T) {
	engine := mustBuildEngine(t, []schemas.RequestOverride{
		{
			Match: "",
			Set:   map[string]interface{}{"new_field": "new_value"},
		},
	})

	input := toJSON(t, map[string]interface{}{"model": "gpt-4o"})
	result, err := engine.ApplyOverrides(input, "gpt-4o", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	parsed := parseJSON(t, result)
	if parsed["new_field"] != "new_value" {
		t.Fatalf("expected new_field=new_value, got %v", parsed["new_field"])
	}
}

// --- Remove operation tests ---

func TestApplyOverrides_RemoveExisting(t *testing.T) {
	engine := mustBuildEngine(t, []schemas.RequestOverride{
		{
			Match:  "",
			Remove: []string{"temperature", "top_p"},
		},
	})

	input := toJSON(t, map[string]interface{}{
		"model":       "gpt-4o",
		"temperature": 0.7,
		"top_p":       0.9,
		"max_tokens":  100,
	})

	result, err := engine.ApplyOverrides(input, "gpt-4o", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	parsed := parseJSON(t, result)
	if _, exists := parsed["temperature"]; exists {
		t.Fatal("temperature should be removed")
	}
	if _, exists := parsed["top_p"]; exists {
		t.Fatal("top_p should be removed")
	}
	if parsed["max_tokens"] != float64(100) {
		t.Fatalf("max_tokens should be preserved, got %v", parsed["max_tokens"])
	}
}

func TestApplyOverrides_RemoveNonExistent(t *testing.T) {
	engine := mustBuildEngine(t, []schemas.RequestOverride{
		{
			Match:  "",
			Remove: []string{"nonexistent"},
		},
	})

	input := toJSON(t, map[string]interface{}{"model": "gpt-4o"})
	result, err := engine.ApplyOverrides(input, "gpt-4o", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Should be unchanged
	parsed := parseJSON(t, result)
	if parsed["model"] != "gpt-4o" {
		t.Fatal("model should be preserved")
	}
}

// --- Defaults operation tests ---

func TestApplyOverrides_DefaultsOnlyWhenAbsent(t *testing.T) {
	engine := mustBuildEngine(t, []schemas.RequestOverride{
		{
			Match:    "",
			Defaults: map[string]interface{}{"temperature": 0.7, "max_tokens": 4096},
		},
	})

	// temperature exists, max_tokens doesn't
	input := toJSON(t, map[string]interface{}{
		"model":       "gpt-4o",
		"temperature": 0.3,
	})

	result, err := engine.ApplyOverrides(input, "gpt-4o", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	parsed := parseJSON(t, result)
	// temperature should keep original
	if parsed["temperature"] != 0.3 {
		t.Fatalf("expected temperature=0.3 (not overwritten), got %v", parsed["temperature"])
	}
	// max_tokens should be set
	if parsed["max_tokens"] != float64(4096) {
		t.Fatalf("expected max_tokens=4096, got %v", parsed["max_tokens"])
	}
}

func TestApplyOverrides_DefaultsAllPresent(t *testing.T) {
	engine := mustBuildEngine(t, []schemas.RequestOverride{
		{
			Match:    "",
			Defaults: map[string]interface{}{"temperature": 0.7},
		},
	})

	input := toJSON(t, map[string]interface{}{
		"model":       "gpt-4o",
		"temperature": 0.3,
	})

	result, err := engine.ApplyOverrides(input, "gpt-4o", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	parsed := parseJSON(t, result)
	if parsed["temperature"] != 0.3 {
		t.Fatalf("expected temperature=0.3 unchanged, got %v", parsed["temperature"])
	}
}

// --- Multiple rules tests ---

func TestApplyOverrides_MultipleRulesAllApplied(t *testing.T) {
	engine := mustBuildEngine(t, []schemas.RequestOverride{
		{
			Match: "model.contains('claude')",
			Set:   map[string]interface{}{"group": "claude-guan"},
		},
		{
			Match:    "",
			Defaults: map[string]interface{}{"temperature": 0.5},
		},
		{
			Match:  "model.contains('claude')",
			Remove: []string{"unwanted"},
		},
	})

	input := toJSON(t, map[string]interface{}{
		"model":    "claude-3-opus",
		"unwanted": "bye",
	})

	result, err := engine.ApplyOverrides(input, "claude-3-opus", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	parsed := parseJSON(t, result)
	if parsed["group"] != "claude-guan" {
		t.Fatalf("expected group=claude-guan, got %v", parsed["group"])
	}
	if parsed["temperature"] != 0.5 {
		t.Fatalf("expected temperature=0.5, got %v", parsed["temperature"])
	}
	if _, exists := parsed["unwanted"]; exists {
		t.Fatal("unwanted should be removed")
	}
}

func TestApplyOverrides_LaterRuleOverridesPrevious(t *testing.T) {
	engine := mustBuildEngine(t, []schemas.RequestOverride{
		{
			Match: "",
			Set:   map[string]interface{}{"temperature": 0.5},
		},
		{
			Match: "model.contains('claude')",
			Set:   map[string]interface{}{"temperature": 0.9},
		},
	})

	input := toJSON(t, map[string]interface{}{"model": "claude-3-opus"})
	result, err := engine.ApplyOverrides(input, "claude-3-opus", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	parsed := parseJSON(t, result)
	if parsed["temperature"] != 0.9 {
		t.Fatalf("expected temperature=0.9 from later rule, got %v", parsed["temperature"])
	}
}

// --- Combined operations in single rule ---

func TestApplyOverrides_CombinedSetRemoveDefaults(t *testing.T) {
	engine := mustBuildEngine(t, []schemas.RequestOverride{
		{
			Match:    "",
			Set:      map[string]interface{}{"forced": "yes"},
			Remove:   []string{"secret"},
			Defaults: map[string]interface{}{"temperature": 0.7, "forced": "no"},
		},
	})

	input := toJSON(t, map[string]interface{}{
		"model":  "gpt-4o",
		"secret": "remove_me",
	})

	result, err := engine.ApplyOverrides(input, "gpt-4o", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	parsed := parseJSON(t, result)
	// defaults applied first: temperature=0.7 (new), forced="no" (new)
	// set applied second: forced="yes" (overwrites defaults)
	// remove applied last: secret removed
	if parsed["forced"] != "yes" {
		t.Fatalf("expected forced=yes (set overrides default), got %v", parsed["forced"])
	}
	if parsed["temperature"] != 0.7 {
		t.Fatalf("expected temperature=0.7 from defaults, got %v", parsed["temperature"])
	}
	if _, exists := parsed["secret"]; exists {
		t.Fatal("secret should be removed")
	}
}

// --- Nil engine safety ---

func TestApplyOverrides_NilEngine(t *testing.T) {
	var engine *RequestOverrideEngine
	input := []byte(`{"model":"gpt-4o"}`)
	result, err := engine.ApplyOverrides(input, "gpt-4o", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(result) != string(input) {
		t.Fatal("nil engine should return body unchanged")
	}
}

// --- ExtractModelFromJSON ---

func TestExtractModelFromJSON(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{"standard", `{"model":"gpt-4o","temperature":0.7}`, "gpt-4o"},
		{"empty", `{"temperature":0.7}`, ""},
		{"nested_safe", `{"model":"claude-3","config":{"model":"inner"}}`, "claude-3"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ExtractModelFromJSON([]byte(tt.input))
			if got != tt.expected {
				t.Fatalf("expected %q, got %q", tt.expected, got)
			}
		})
	}
}

// --- Set with nested objects ---

func TestApplyOverrides_SetNestedObject(t *testing.T) {
	engine := mustBuildEngine(t, []schemas.RequestOverride{
		{
			Match: "",
			Set: map[string]interface{}{
				"metadata": map[string]interface{}{
					"env":  "production",
					"team": "ai",
				},
			},
		},
	})

	input := toJSON(t, map[string]interface{}{"model": "gpt-4o"})
	result, err := engine.ApplyOverrides(input, "gpt-4o", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	parsed := parseJSON(t, result)
	metadata, ok := parsed["metadata"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected metadata to be an object, got %T", parsed["metadata"])
	}
	if metadata["env"] != "production" || metadata["team"] != "ai" {
		t.Fatalf("unexpected metadata: %v", metadata)
	}
}

// --- Helpers ---

func mustBuildEngine(t *testing.T, overrides []schemas.RequestOverride) *RequestOverrideEngine {
	t.Helper()
	engine, err := NewRequestOverrideEngine(overrides)
	if err != nil {
		t.Fatalf("failed to build engine: %v", err)
	}
	return engine
}

func toJSON(t *testing.T, v interface{}) []byte {
	t.Helper()
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		t.Fatalf("failed to marshal JSON: %v", err)
	}
	return data
}

func parseJSON(t *testing.T, data []byte) map[string]interface{} {
	t.Helper()
	var result map[string]interface{}
	if err := json.Unmarshal(data, &result); err != nil {
		t.Fatalf("failed to parse JSON: %v\ndata: %s", err, string(data))
	}
	return result
}
