package gmi

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/bytedance/sonic"

	"github.com/maximhq/bifrost/core/providers/gemini"
	schemas "github.com/maximhq/bifrost/core/schemas"
)

type testLogger struct{}

func (testLogger) Debug(string, ...any)                   {}
func (testLogger) Info(string, ...any)                    {}
func (testLogger) Warn(string, ...any)                    {}
func (testLogger) Error(string, ...any)                   {}
func (testLogger) Fatal(string, ...any)                   {}
func (testLogger) SetLevel(schemas.LogLevel)              {}
func (testLogger) SetOutputType(schemas.LoggerOutputType) {}
func (testLogger) LogHTTPRequest(schemas.LogLevel, string) schemas.LogEventBuilder {
	return schemas.NoopLogEvent
}

func newTestProvider(t *testing.T, baseURL string) *GMIProvider {
	t.Helper()
	provider, err := NewGMIProvider(&schemas.ProviderConfig{
		NetworkConfig: schemas.NetworkConfig{
			BaseURL:                        baseURL,
			DefaultRequestTimeoutInSeconds: 5,
		},
	}, testLogger{})
	if err != nil {
		t.Fatalf("NewGMIProvider() error = %v", err)
	}
	return provider
}

func newTestCustomProvider(t *testing.T, baseURL string, providerKey schemas.ModelProvider, overrides map[schemas.RequestType]string) *GMIProvider {
	t.Helper()
	provider, err := NewGMIProvider(&schemas.ProviderConfig{
		NetworkConfig: schemas.NetworkConfig{
			BaseURL:                        baseURL,
			DefaultRequestTimeoutInSeconds: 5,
		},
		CustomProviderConfig: &schemas.CustomProviderConfig{
			CustomProviderKey:    string(providerKey),
			BaseProviderType:     schemas.GMI,
			RequestPathOverrides: overrides,
		},
	}, testLogger{})
	if err != nil {
		t.Fatalf("NewGMIProvider() error = %v", err)
	}
	return provider
}

func newTestContext() *schemas.BifrostContext {
	return schemas.NewBifrostContext(context.Background(), schemas.NoDeadline)
}

func makeChatRequest(model string) *schemas.BifrostChatRequest {
	return &schemas.BifrostChatRequest{
		Provider: schemas.GMI,
		Model:    model,
		Input: []schemas.ChatMessage{
			{
				Role: schemas.ChatMessageRoleUser,
				Content: &schemas.ChatMessageContent{
					ContentStr: schemas.Ptr("hello"),
				},
			},
		},
		Params: &schemas.ChatParameters{
			MaxCompletionTokens: schemas.Ptr(32),
		},
	}
}

func makeResponsesRequest(model string) *schemas.BifrostResponsesRequest {
	return &schemas.BifrostResponsesRequest{
		Provider: schemas.GMI,
		Model:    model,
		Input: []schemas.ResponsesMessage{
			{
				Role: schemas.Ptr(schemas.ResponsesInputMessageRoleUser),
				Content: &schemas.ResponsesMessageContent{
					ContentStr: schemas.Ptr("hello"),
				},
			},
		},
		Params: &schemas.ResponsesParameters{
			MaxOutputTokens: schemas.Ptr(32),
		},
	}
}

func noOpPostHookRunner(_ *schemas.BifrostContext, result *schemas.BifrostResponse, err *schemas.BifrostError) (*schemas.BifrostResponse, *schemas.BifrostError) {
	return result, err
}

func TestResolveModel(t *testing.T) {
	t.Parallel()

	tests := []struct {
		model            string
		wantFamily       modelFamily
		wantUpstream     string
		wantDeployment   string
		wantErrorMessage string
	}{
		{model: "openai/gpt-5.4", wantFamily: modelFamilyResponses, wantUpstream: "openai/gpt-5.4", wantDeployment: "openai/gpt-5.4"},
		{model: "deepseek-ai/DeepSeek-R1", wantFamily: modelFamilyChatCompletions, wantUpstream: "deepseek-ai/DeepSeek-R1", wantDeployment: "deepseek-ai/DeepSeek-R1"},
		{model: "anthropic/claude-sonnet-4.6", wantFamily: modelFamilyAnthropic, wantUpstream: "anthropic/claude-sonnet-4.6", wantDeployment: "anthropic/claude-sonnet-4.6"},
		{model: "google/gemini-3.1-flash-lite-preview", wantFamily: modelFamilyGoogle, wantUpstream: "gemini-3.1-flash-lite-preview", wantDeployment: "gemini-3.1-flash-lite-preview"},
		{model: "gpt-5.4", wantErrorMessage: "gmi model must be in family/model format"},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.model, func(t *testing.T) {
			t.Parallel()

			got, err := resolveModel(tt.model, schemas.GMI)
			if tt.wantErrorMessage != "" {
				if err == nil || err.Error == nil || !strings.Contains(err.Error.Message, tt.wantErrorMessage) {
					t.Fatalf("resolveModel(%q) error = %#v, want substring %q", tt.model, err, tt.wantErrorMessage)
				}
				return
			}

			if err != nil {
				t.Fatalf("resolveModel(%q) unexpected error = %v", tt.model, err)
			}
			if got.family != tt.wantFamily || got.upstreamModel != tt.wantUpstream || got.modelDeployment != tt.wantDeployment {
				t.Fatalf("resolveModel(%q) = %+v, want family=%q upstream=%q deployment=%q", tt.model, got, tt.wantFamily, tt.wantUpstream, tt.wantDeployment)
			}
		})
	}
}

func TestResponsesAnthropicKeepsBetaHeadersOnNonStreamRequests(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("anthropic-beta"); !strings.Contains(got, "fast-mode-2025-02-19") {
			t.Fatalf("anthropic-beta = %q, want to contain fast-mode-2025-02-19", got)
		}
		_ = sonic.ConfigDefault.NewEncoder(w).Encode(map[string]any{
			"id":          "msg_1",
			"type":        "message",
			"role":        "assistant",
			"model":       "anthropic/claude-sonnet-4.6",
			"content":     []map[string]any{{"type": "text", "text": "hello"}},
			"stop_reason": "end_turn",
		})
	}))
	defer server.Close()

	ctx := newTestContext()
	ctx.SetValue(schemas.BifrostContextKeyExtraHeaders, map[string][]string{
		"anthropic-beta": {"fast-mode-2025-02-19"},
	})

	provider := newTestProvider(t, server.URL)
	_, err := provider.Responses(ctx, schemas.Key{Value: schemas.EnvVar{Val: "test-key"}}, makeResponsesRequest("anthropic/claude-sonnet-4.6"))
	if err != nil {
		t.Fatalf("Responses() error = %v", err)
	}
}

func TestCustomProviderConfigUsesCustomKeyAndOverrides(t *testing.T) {
	t.Parallel()

	const customProviderKey = schemas.ModelProvider("custom-gmi")
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/custom-responses":
			response := &schemas.BifrostChatResponse{
				ID:      "chatcmpl_custom",
				Object:  "chat.completion",
				Created: 123,
				Model:   "deepseek-ai/DeepSeek-R1",
				Choices: []schemas.BifrostResponseChoice{
					{
						Index: 0,
						ChatNonStreamResponseChoice: &schemas.ChatNonStreamResponseChoice{
							Message: &schemas.ChatMessage{
								Role: schemas.ChatMessageRoleAssistant,
								Content: &schemas.ChatMessageContent{
									ContentStr: schemas.Ptr("custom provider response"),
								},
							},
						},
						FinishReason: schemas.Ptr(string(schemas.BifrostFinishReasonStop)),
					},
				},
			}
			_ = sonic.ConfigDefault.NewEncoder(w).Encode(response)
		case "/v1/models":
			_ = sonic.ConfigDefault.NewEncoder(w).Encode(map[string]any{
				"object": "list",
				"data": []map[string]any{
					{"id": "deepseek-ai/DeepSeek-R1", "object": "model", "owned_by": "deepseek-ai"},
				},
			})
		default:
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
	}))
	defer server.Close()

	provider := newTestCustomProvider(t, server.URL, customProviderKey, map[schemas.RequestType]string{
		schemas.ResponsesRequest: "/custom-responses",
	})
	if got := provider.GetProviderKey(); got != customProviderKey {
		t.Fatalf("GetProviderKey() = %q, want %q", got, customProviderKey)
	}

	resp, err := provider.Responses(newTestContext(), schemas.Key{Value: schemas.EnvVar{Val: "test-key"}}, makeResponsesRequest("deepseek-ai/DeepSeek-R1"))
	if err != nil {
		t.Fatalf("Responses() error = %v", err)
	}
	if resp.ExtraFields.Provider != customProviderKey {
		t.Fatalf("provider = %q, want %q", resp.ExtraFields.Provider, customProviderKey)
	}
	if len(resp.Output) == 0 || resp.Output[0].Content == nil || len(resp.Output[0].Content.ContentBlocks) == 0 || resp.Output[0].Content.ContentBlocks[0].Text == nil || *resp.Output[0].Content.ContentBlocks[0].Text != "custom provider response" {
		t.Fatalf("Responses() returned unexpected output: %#v", resp.Output)
	}

	listResp, err := provider.ListModels(newTestContext(), []schemas.Key{{Value: schemas.EnvVar{Val: "test-key"}}}, &schemas.BifrostListModelsRequest{
		Provider: customProviderKey,
	})
	if err != nil {
		t.Fatalf("ListModels() error = %v", err)
	}
	if len(listResp.Data) != 1 || listResp.Data[0].ID != "custom-gmi/deepseek-ai/DeepSeek-R1" {
		t.Fatalf("ListModels() returned %#v, want custom provider-prefixed model ID", listResp.Data)
	}
}

func TestListModelsNormalizesProviderPrefix(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		_ = sonic.ConfigDefault.NewEncoder(w).Encode(map[string]any{
			"object": "list",
			"data": []map[string]any{
				{"id": "openai/gpt-5.4", "object": "model", "owned_by": "openai"},
				{"id": "gemini-3.1-flash-lite-preview", "object": "model", "owned_by": "google"},
			},
		})
	}))
	defer server.Close()

	provider := newTestProvider(t, server.URL)
	resp, err := provider.ListModels(newTestContext(), []schemas.Key{{Value: schemas.EnvVar{Val: "test-key"}}}, &schemas.BifrostListModelsRequest{
		Provider: schemas.GMI,
	})
	if err != nil {
		t.Fatalf("ListModels() error = %v", err)
	}
	if len(resp.Data) != 2 {
		t.Fatalf("ListModels() returned %d models, want 2", len(resp.Data))
	}
	if resp.Data[0].ID != "gmi/openai/gpt-5.4" {
		t.Fatalf("first model ID = %q, want %q", resp.Data[0].ID, "gmi/openai/gpt-5.4")
	}
	if resp.Data[1].ID != "gmi/google/gemini-3.1-flash-lite-preview" {
		t.Fatalf("second model ID = %q, want %q", resp.Data[1].ID, "gmi/google/gemini-3.1-flash-lite-preview")
	}
}

func TestChatCompletionOpenAIRoutesToResponses(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"model": "openai/gpt-5.4"`) {
			t.Fatalf("request body missing model: %s", body)
		}

		response := &schemas.BifrostResponsesResponse{
			ID:        schemas.Ptr("resp_1"),
			Object:    "response",
			CreatedAt: 123,
			Model:     "openai/gpt-5.4",
			Output: []schemas.ResponsesMessage{
				{
					ID:   schemas.Ptr("msg_1"),
					Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage),
					Role: schemas.Ptr(schemas.ResponsesInputMessageRoleAssistant),
					Content: &schemas.ResponsesMessageContent{
						ContentBlocks: []schemas.ResponsesMessageContentBlock{
							{
								Type:                              schemas.ResponsesOutputMessageContentTypeText,
								Text:                              schemas.Ptr("hello from responses"),
								ResponsesOutputMessageContentText: &schemas.ResponsesOutputMessageContentText{},
							},
						},
					},
				},
			},
			StopReason: schemas.Ptr("stop"),
			Usage: &schemas.ResponsesResponseUsage{
				InputTokens:  2,
				OutputTokens: 3,
				TotalTokens:  5,
			},
		}
		_ = sonic.ConfigDefault.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := newTestProvider(t, server.URL)
	resp, err := provider.ChatCompletion(newTestContext(), schemas.Key{Value: schemas.EnvVar{Val: "test-key"}}, makeChatRequest("openai/gpt-5.4"))
	if err != nil {
		t.Fatalf("ChatCompletion() error = %v", err)
	}
	if resp.ExtraFields.Provider != schemas.GMI {
		t.Fatalf("provider = %q, want %q", resp.ExtraFields.Provider, schemas.GMI)
	}
	if len(resp.Choices) == 0 || resp.Choices[0].ChatNonStreamResponseChoice == nil || resp.Choices[0].ChatNonStreamResponseChoice.Message == nil {
		t.Fatalf("ChatCompletion() returned empty choices: %#v", resp.Choices)
	}
	message := resp.Choices[0].ChatNonStreamResponseChoice.Message
	var got *string
	if message.Content != nil {
		if message.Content.ContentStr != nil {
			got = message.Content.ContentStr
		} else if len(message.Content.ContentBlocks) > 0 {
			got = message.Content.ContentBlocks[0].Text
		}
	}
	if got == nil || *got != "hello from responses" {
		t.Fatalf("content = %v, want %q", got, "hello from responses")
	}
	if resp.Choices[0].FinishReason == nil || *resp.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish_reason = %v, want %q", resp.Choices[0].FinishReason, "stop")
	}
}

func TestChatCompletionOpenAIResponsesPreservesChatSpecificParameters(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}

		var payload struct {
			Model string   `json:"model"`
			User  string   `json:"user"`
			Stop  []string `json:"stop"`
			Text  struct {
				Format struct {
					Type   string `json:"type"`
					Schema struct {
						Name   string                 `json:"name"`
						Schema map[string]interface{} `json:"schema"`
						Strict bool                   `json:"strict"`
					} `json:"schema"`
				} `json:"format"`
			} `json:"text"`
		}
		if err := sonic.ConfigDefault.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("failed to decode request body: %v", err)
		}
		if payload.Model != "openai/gpt-5.4" {
			t.Fatalf("model = %q, want %q", payload.Model, "openai/gpt-5.4")
		}
		if payload.User != "chat-user" {
			t.Fatalf("user = %q, want %q", payload.User, "chat-user")
		}
		if len(payload.Stop) != 1 || payload.Stop[0] != "END" {
			t.Fatalf("stop = %#v, want [\"END\"]", payload.Stop)
		}
		if payload.Text.Format.Type != "json_schema" {
			t.Fatalf("text.format.type = %q, want %q", payload.Text.Format.Type, "json_schema")
		}
		if payload.Text.Format.Schema.Name != "answer" {
			t.Fatalf("text.format.schema.name = %q, want %q", payload.Text.Format.Schema.Name, "answer")
		}
		if !payload.Text.Format.Schema.Strict {
			t.Fatal("text.format.schema.strict = false, want true")
		}
		if properties, ok := payload.Text.Format.Schema.Schema["properties"].(map[string]interface{}); !ok || properties["answer"] == nil {
			t.Fatalf("text.format.schema.schema.properties missing answer: %#v", payload.Text.Format.Schema.Schema)
		}

		response := &schemas.BifrostResponsesResponse{
			ID:        schemas.Ptr("resp_chat_params"),
			Object:    "response",
			CreatedAt: 123,
			Model:     "openai/gpt-5.4",
			Output: []schemas.ResponsesMessage{
				{
					ID:   schemas.Ptr("msg_1"),
					Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage),
					Role: schemas.Ptr(schemas.ResponsesInputMessageRoleAssistant),
					Content: &schemas.ResponsesMessageContent{
						ContentBlocks: []schemas.ResponsesMessageContentBlock{
							{
								Type:                              schemas.ResponsesOutputMessageContentTypeText,
								Text:                              schemas.Ptr("ok"),
								ResponsesOutputMessageContentText: &schemas.ResponsesOutputMessageContentText{},
							},
						},
					},
				},
			},
			StopReason: schemas.Ptr("stop"),
		}
		_ = sonic.ConfigDefault.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	request := makeChatRequest("openai/gpt-5.4")
	request.Params.User = schemas.Ptr("chat-user")
	request.Params.Stop = []string{"END"}
	responseFormat := interface{}(map[string]interface{}{
		"type": "json_schema",
		"json_schema": map[string]interface{}{
			"name":   "answer",
			"strict": true,
			"schema": map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"answer": map[string]interface{}{
						"type": "string",
					},
				},
				"required": []string{"answer"},
			},
		},
	})
	request.Params.ResponseFormat = &responseFormat

	provider := newTestProvider(t, server.URL)
	if _, err := provider.ChatCompletion(newTestContext(), schemas.Key{Value: schemas.EnvVar{Val: "test-key"}}, request); err != nil {
		t.Fatalf("ChatCompletion() error = %v", err)
	}
}

func TestChatCompletionOpenAICompatibleRoutesToChatCompletions(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"model": "deepseek-ai/DeepSeek-R1"`) {
			t.Fatalf("request body missing model: %s", body)
		}

		response := &schemas.BifrostChatResponse{
			ID:      "chatcmpl_1",
			Object:  "chat.completion",
			Created: 123,
			Model:   "deepseek-ai/DeepSeek-R1",
			Choices: []schemas.BifrostResponseChoice{
				{
					Index: 0,
					ChatNonStreamResponseChoice: &schemas.ChatNonStreamResponseChoice{
						Message: &schemas.ChatMessage{
							Role: schemas.ChatMessageRoleAssistant,
							Content: &schemas.ChatMessageContent{
								ContentStr: schemas.Ptr("hello from chat completions"),
							},
						},
					},
					FinishReason: schemas.Ptr(string(schemas.BifrostFinishReasonStop)),
				},
			},
			Usage: &schemas.BifrostLLMUsage{
				PromptTokens:     2,
				CompletionTokens: 3,
				TotalTokens:      5,
			},
		}
		_ = sonic.ConfigDefault.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := newTestProvider(t, server.URL)
	resp, err := provider.ChatCompletion(newTestContext(), schemas.Key{Value: schemas.EnvVar{Val: "test-key"}}, makeChatRequest("deepseek-ai/DeepSeek-R1"))
	if err != nil {
		t.Fatalf("ChatCompletion() error = %v", err)
	}
	if len(resp.Choices) == 0 || resp.Choices[0].ChatNonStreamResponseChoice == nil || resp.Choices[0].ChatNonStreamResponseChoice.Message == nil {
		t.Fatalf("ChatCompletion() returned empty choices: %#v", resp.Choices)
	}
	if resp.Choices[0].ChatNonStreamResponseChoice.Message.Content == nil || resp.Choices[0].ChatNonStreamResponseChoice.Message.Content.ContentStr == nil || *resp.Choices[0].ChatNonStreamResponseChoice.Message.Content.ContentStr != "hello from chat completions" {
		t.Fatalf("unexpected message: %#v", resp.Choices[0].ChatNonStreamResponseChoice.Message)
	}
}

func TestResponsesOpenAICompatibleRoutesToChatCompletions(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"model": "deepseek-ai/DeepSeek-R1"`) {
			t.Fatalf("request body missing model: %s", body)
		}

		response := &schemas.BifrostChatResponse{
			ID:      "chatcmpl_resp",
			Object:  "chat.completion",
			Created: 123,
			Model:   "deepseek-ai/DeepSeek-R1",
			Choices: []schemas.BifrostResponseChoice{
				{
					Index: 0,
					ChatNonStreamResponseChoice: &schemas.ChatNonStreamResponseChoice{
						Message: &schemas.ChatMessage{
							Role: schemas.ChatMessageRoleAssistant,
							Content: &schemas.ChatMessageContent{
								ContentStr: schemas.Ptr("responses through chat completions"),
							},
						},
					},
					FinishReason: schemas.Ptr(string(schemas.BifrostFinishReasonStop)),
				},
			},
			Usage: &schemas.BifrostLLMUsage{
				PromptTokens:     2,
				CompletionTokens: 3,
				TotalTokens:      5,
			},
		}
		_ = sonic.ConfigDefault.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	provider := newTestProvider(t, server.URL)
	resp, err := provider.Responses(newTestContext(), schemas.Key{Value: schemas.EnvVar{Val: "test-key"}}, makeResponsesRequest("deepseek-ai/DeepSeek-R1"))
	if err != nil {
		t.Fatalf("Responses() error = %v", err)
	}
	if len(resp.Output) == 0 || resp.Output[0].Content == nil || len(resp.Output[0].Content.ContentBlocks) == 0 || resp.Output[0].Content.ContentBlocks[0].Text == nil || *resp.Output[0].Content.ContentBlocks[0].Text != "responses through chat completions" {
		t.Fatalf("unexpected output: %#v", resp.Output)
	}
	if resp.Usage == nil || resp.Usage.TotalTokens != 5 {
		t.Fatalf("unexpected usage: %#v", resp.Usage)
	}
}

func TestResponsesOpenAICompatiblePreservesResponseSpecificParameters(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}

		var payload struct {
			Model          string `json:"model"`
			User           string `json:"user"`
			ResponseFormat struct {
				Type       string `json:"type"`
				JSONSchema struct {
					Name   string                 `json:"name"`
					Schema map[string]interface{} `json:"schema"`
					Strict bool                   `json:"strict"`
				} `json:"json_schema"`
			} `json:"response_format"`
		}
		if err := sonic.ConfigDefault.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("failed to decode request body: %v", err)
		}
		if payload.Model != "deepseek-ai/DeepSeek-R1" {
			t.Fatalf("model = %q, want %q", payload.Model, "deepseek-ai/DeepSeek-R1")
		}
		if payload.User != "responses-user" {
			t.Fatalf("user = %q, want %q", payload.User, "responses-user")
		}
		if payload.ResponseFormat.Type != "json_schema" {
			t.Fatalf("response_format.type = %q, want %q", payload.ResponseFormat.Type, "json_schema")
		}
		if payload.ResponseFormat.JSONSchema.Name != "answer" {
			t.Fatalf("response_format.json_schema.name = %q, want %q", payload.ResponseFormat.JSONSchema.Name, "answer")
		}
		if !payload.ResponseFormat.JSONSchema.Strict {
			t.Fatal("response_format.json_schema.strict = false, want true")
		}
		if properties, ok := payload.ResponseFormat.JSONSchema.Schema["properties"].(map[string]interface{}); !ok || properties["answer"] == nil {
			t.Fatalf("response_format.json_schema.schema.properties missing answer: %#v", payload.ResponseFormat.JSONSchema.Schema)
		}

		response := &schemas.BifrostChatResponse{
			ID:      "chatcmpl_resp_params",
			Object:  "chat.completion",
			Created: 123,
			Model:   "deepseek-ai/DeepSeek-R1",
			Choices: []schemas.BifrostResponseChoice{
				{
					Index: 0,
					ChatNonStreamResponseChoice: &schemas.ChatNonStreamResponseChoice{
						Message: &schemas.ChatMessage{
							Role: schemas.ChatMessageRoleAssistant,
							Content: &schemas.ChatMessageContent{
								ContentStr: schemas.Ptr("ok"),
							},
						},
					},
					FinishReason: schemas.Ptr(string(schemas.BifrostFinishReasonStop)),
				},
			},
		}
		_ = sonic.ConfigDefault.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	request := makeResponsesRequest("deepseek-ai/DeepSeek-R1")
	request.Params.User = schemas.Ptr("responses-user")
	schema := interface{}(map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"answer": map[string]interface{}{
				"type": "string",
			},
		},
		"required": []string{"answer"},
	})
	request.Params.Text = &schemas.ResponsesTextConfig{
		Format: &schemas.ResponsesTextConfigFormat{
			Type: "json_schema",
			JSONSchema: &schemas.ResponsesTextConfigFormatJSONSchema{
				Name:   schemas.Ptr("answer"),
				Schema: &schema,
				Strict: schemas.Ptr(true),
			},
		},
	}

	provider := newTestProvider(t, server.URL)
	if _, err := provider.Responses(newTestContext(), schemas.Key{Value: schemas.EnvVar{Val: "test-key"}}, request); err != nil {
		t.Fatalf("Responses() error = %v", err)
	}
}

func TestResponsesOpenAICompatiblePreservesTopLevelStructuredOutputMetadata(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}

		var payload struct {
			ResponseFormat struct {
				Type       string `json:"type"`
				JSONSchema struct {
					Name   string `json:"name"`
					Strict bool   `json:"strict"`
				} `json:"json_schema"`
			} `json:"response_format"`
		}
		if err := sonic.ConfigDefault.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("failed to decode request body: %v", err)
		}
		if payload.ResponseFormat.Type != "json_schema" {
			t.Fatalf("response_format.type = %q, want %q", payload.ResponseFormat.Type, "json_schema")
		}
		if payload.ResponseFormat.JSONSchema.Name != "top_level_schema" {
			t.Fatalf("response_format.json_schema.name = %q, want %q", payload.ResponseFormat.JSONSchema.Name, "top_level_schema")
		}
		if !payload.ResponseFormat.JSONSchema.Strict {
			t.Fatal("response_format.json_schema.strict = false, want true")
		}

		response := &schemas.BifrostChatResponse{
			ID:      "chatcmpl_resp_top_level_schema",
			Object:  "chat.completion",
			Created: 123,
			Model:   "deepseek-ai/DeepSeek-R1",
			Choices: []schemas.BifrostResponseChoice{
				{
					Index: 0,
					ChatNonStreamResponseChoice: &schemas.ChatNonStreamResponseChoice{
						Message: &schemas.ChatMessage{
							Role: schemas.ChatMessageRoleAssistant,
							Content: &schemas.ChatMessageContent{
								ContentStr: schemas.Ptr("ok"),
							},
						},
					},
					FinishReason: schemas.Ptr(string(schemas.BifrostFinishReasonStop)),
				},
			},
		}
		_ = sonic.ConfigDefault.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	schema := interface{}(map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"answer": map[string]interface{}{
				"type": "string",
			},
		},
		"required": []string{"answer"},
	})
	request := makeResponsesRequest("deepseek-ai/DeepSeek-R1")
	request.Params.Text = &schemas.ResponsesTextConfig{
		Format: &schemas.ResponsesTextConfigFormat{
			Type:   "json_schema",
			Name:   schemas.Ptr("top_level_schema"),
			Strict: schemas.Ptr(true),
			JSONSchema: &schemas.ResponsesTextConfigFormatJSONSchema{
				Schema: &schema,
			},
		},
	}

	provider := newTestProvider(t, server.URL)
	if _, err := provider.Responses(newTestContext(), schemas.Key{Value: schemas.EnvVar{Val: "test-key"}}, request); err != nil {
		t.Fatalf("Responses() error = %v", err)
	}
}

func TestResponsesAnthropicRoutesToMessages(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Fatalf("Authorization = %q, want %q", got, "Bearer test-key")
		}
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"model": "anthropic/claude-sonnet-4.6"`) {
			t.Fatalf("request body missing model: %s", body)
		}
		_ = sonic.ConfigDefault.NewEncoder(w).Encode(map[string]any{
			"id":          "msg_1",
			"type":        "message",
			"role":        "assistant",
			"model":       "anthropic/claude-sonnet-4.6",
			"content":     []map[string]any{{"type": "text", "text": "hello from messages"}},
			"stop_reason": "end_turn",
			"usage":       map[string]any{"input_tokens": 2, "output_tokens": 3},
		})
	}))
	defer server.Close()

	provider := newTestProvider(t, server.URL)
	resp, err := provider.Responses(newTestContext(), schemas.Key{Value: schemas.EnvVar{Val: "test-key"}}, makeResponsesRequest("anthropic/claude-sonnet-4.6"))
	if err != nil {
		t.Fatalf("Responses() error = %v", err)
	}
	if resp.ExtraFields.Provider != schemas.GMI {
		t.Fatalf("provider = %q, want %q", resp.ExtraFields.Provider, schemas.GMI)
	}
	if len(resp.Output) == 0 {
		t.Fatalf("Responses() returned empty output")
	}
	if resp.Output[0].Content == nil || len(resp.Output[0].Content.ContentBlocks) == 0 || resp.Output[0].Content.ContentBlocks[0].Text == nil || *resp.Output[0].Content.ContentBlocks[0].Text != "hello from messages" {
		t.Fatalf("unexpected output: %#v", resp.Output[0])
	}
}

func TestResponsesGoogleRoutesToGenerateContent(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		wantPath := "/v1/models/gemini-3.1-flash-lite-preview:generateContent"
		if r.URL.Path != wantPath {
			t.Fatalf("unexpected path %q, want %q", r.URL.Path, wantPath)
		}
		body, _ := io.ReadAll(r.Body)
		if !strings.Contains(string(body), `"model": "gemini-3.1-flash-lite-preview"`) {
			t.Fatalf("request body missing stripped model: %s", body)
		}

		resp := &gemini.GenerateContentResponse{
			ResponseID:   "resp_google",
			ModelVersion: "gemini-3.1-flash-lite-preview",
			Candidates: []*gemini.Candidate{
				{
					FinishReason: gemini.FinishReasonStop,
					Content: &gemini.Content{
						Role: "model",
						Parts: []*gemini.Part{
							{Text: "hello from generateContent"},
						},
					},
				},
			},
			UsageMetadata: &gemini.GenerateContentResponseUsageMetadata{
				PromptTokenCount:     2,
				CandidatesTokenCount: 3,
				TotalTokenCount:      5,
			},
		}
		_ = sonic.ConfigDefault.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	provider := newTestProvider(t, server.URL)
	resp, err := provider.Responses(newTestContext(), schemas.Key{Value: schemas.EnvVar{Val: "test-key"}}, makeResponsesRequest("google/gemini-3.1-flash-lite-preview"))
	if err != nil {
		t.Fatalf("Responses() error = %v", err)
	}
	if resp.ExtraFields.ModelDeployment != "gemini-3.1-flash-lite-preview" {
		t.Fatalf("model_deployment = %q, want %q", resp.ExtraFields.ModelDeployment, "gemini-3.1-flash-lite-preview")
	}
	if len(resp.Output) == 0 || resp.Output[0].Content == nil || len(resp.Output[0].Content.ContentBlocks) == 0 {
		t.Fatalf("unexpected output: %#v", resp.Output)
	}
}

func TestChatCompletionStreamOpenAIResponsesAdapter(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer does not support flushing")
		}

		events := []string{
			`{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","created_at":123,"model":"openai/gpt-5.4"}}`,
			`{"type":"response.output_item.added","sequence_number":1,"output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","content":[]}}`,
			`{"type":"response.output_text.delta","sequence_number":2,"output_index":0,"item_id":"msg_1","content_index":0,"delta":"hel"}`,
			`{"type":"response.output_text.delta","sequence_number":3,"output_index":0,"item_id":"msg_1","content_index":0,"delta":"lo"}`,
			`{"type":"response.completed","sequence_number":4,"response":{"id":"resp_1","created_at":123,"model":"openai/gpt-5.4","stop_reason":"stop","usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}`,
		}

		for _, event := range events {
			fmt.Fprintf(w, "data: %s\n\n", event)
			flusher.Flush()
		}
	}))
	defer server.Close()

	provider := newTestProvider(t, server.URL)
	stream, err := provider.ChatCompletionStream(newTestContext(), noOpPostHookRunner, schemas.Key{Value: schemas.EnvVar{Val: "test-key"}}, makeChatRequest("openai/gpt-5.4"))
	if err != nil {
		t.Fatalf("ChatCompletionStream() error = %v", err)
	}

	var gotRole bool
	var content strings.Builder
	var final *schemas.BifrostChatResponse
	timeout := time.After(5 * time.Second)

	for {
		select {
		case chunk, ok := <-stream:
			if !ok {
				if !gotRole {
					t.Fatal("stream never emitted assistant role chunk")
				}
				if content.String() != "hello" {
					t.Fatalf("content = %q, want %q", content.String(), "hello")
				}
				if final == nil {
					t.Fatal("stream never emitted final chunk")
				}
				if final.Choices[0].FinishReason == nil || *final.Choices[0].FinishReason != "stop" {
					t.Fatalf("final finish_reason = %v, want %q", final.Choices[0].FinishReason, "stop")
				}
				if final.Usage == nil || final.Usage.TotalTokens != 5 {
					t.Fatalf("final usage = %#v, want total_tokens=5", final.Usage)
				}
				return
			}
			if chunk.BifrostError != nil {
				t.Fatalf("unexpected stream error: %#v", chunk.BifrostError)
			}
			if chunk.BifrostChatResponse == nil || len(chunk.BifrostChatResponse.Choices) == 0 || chunk.BifrostChatResponse.Choices[0].ChatStreamResponseChoice == nil {
				continue
			}
			delta := chunk.BifrostChatResponse.Choices[0].ChatStreamResponseChoice.Delta
			if delta == nil {
				continue
			}
			if delta.Role != nil && *delta.Role == "assistant" {
				gotRole = true
			}
			if delta.Content != nil {
				content.WriteString(*delta.Content)
			}
			if chunk.BifrostChatResponse.Choices[0].FinishReason != nil {
				final = chunk.BifrostChatResponse
			}
		case <-timeout:
			t.Fatal("timed out waiting for stream")
		}
	}
}

func TestChatCompletionStreamOpenAIResponsesPreservesChatSpecificParameters(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}

		var payload struct {
			User string   `json:"user"`
			Stop []string `json:"stop"`
			Text struct {
				Format struct {
					Type string `json:"type"`
				} `json:"format"`
			} `json:"text"`
		}
		if err := sonic.ConfigDefault.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("failed to decode request body: %v", err)
		}
		if payload.User != "stream-user" {
			t.Fatalf("user = %q, want %q", payload.User, "stream-user")
		}
		if len(payload.Stop) != 1 || payload.Stop[0] != "END" {
			t.Fatalf("stop = %#v, want [\"END\"]", payload.Stop)
		}
		if payload.Text.Format.Type != "json_object" {
			t.Fatalf("text.format.type = %q, want %q", payload.Text.Format.Type, "json_object")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer does not support flushing")
		}

		events := []string{
			`{"type":"response.created","sequence_number":0,"response":{"id":"resp_stream_params","created_at":123,"model":"openai/gpt-5.4"}}`,
			`{"type":"response.completed","sequence_number":1,"response":{"id":"resp_stream_params","created_at":123,"model":"openai/gpt-5.4","stop_reason":"stop","usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}`,
		}

		for _, event := range events {
			fmt.Fprintf(w, "data: %s\n\n", event)
			flusher.Flush()
		}
	}))
	defer server.Close()

	request := makeChatRequest("openai/gpt-5.4")
	request.Params.User = schemas.Ptr("stream-user")
	request.Params.Stop = []string{"END"}
	responseFormat := interface{}(map[string]interface{}{"type": "json_object"})
	request.Params.ResponseFormat = &responseFormat

	provider := newTestProvider(t, server.URL)
	stream, err := provider.ChatCompletionStream(newTestContext(), noOpPostHookRunner, schemas.Key{Value: schemas.EnvVar{Val: "test-key"}}, request)
	if err != nil {
		t.Fatalf("ChatCompletionStream() error = %v", err)
	}

	timeout := time.After(5 * time.Second)
	for {
		select {
		case chunk, ok := <-stream:
			if !ok {
				return
			}
			if chunk.BifrostError != nil {
				t.Fatalf("unexpected stream error: %#v", chunk.BifrostError)
			}
		case <-timeout:
			t.Fatal("timed out waiting for stream")
		}
	}
}

func TestChatCompletionStreamOpenAIResponsesUsesNestedFailedError(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer does not support flushing")
		}

		events := []string{
			`{"type":"response.created","sequence_number":0,"response":{"id":"resp_failed","created_at":123,"model":"openai/gpt-5.4"}}`,
			`{"type":"response.failed","sequence_number":1,"response":{"id":"resp_failed","created_at":123,"model":"openai/gpt-5.4","error":{"message":"nested failure reason","code":"nested_error"}}}`,
		}

		for _, event := range events {
			fmt.Fprintf(w, "data: %s\n\n", event)
			flusher.Flush()
		}
	}))
	defer server.Close()

	provider := newTestProvider(t, server.URL)
	stream, err := provider.ChatCompletionStream(newTestContext(), noOpPostHookRunner, schemas.Key{Value: schemas.EnvVar{Val: "test-key"}}, makeChatRequest("openai/gpt-5.4"))
	if err != nil {
		t.Fatalf("ChatCompletionStream() error = %v", err)
	}

	timeout := time.After(5 * time.Second)
	for {
		select {
		case chunk, ok := <-stream:
			if !ok {
				t.Fatal("stream closed before emitting error")
			}
			if chunk.BifrostError == nil {
				continue
			}
			if chunk.BifrostError.Error == nil {
				t.Fatalf("stream error missing error field: %#v", chunk.BifrostError)
			}
			if chunk.BifrostError.Error.Message != "nested failure reason" {
				t.Fatalf("error message = %q, want %q", chunk.BifrostError.Error.Message, "nested failure reason")
			}
			if chunk.BifrostError.Error.Code == nil || *chunk.BifrostError.Error.Code != "nested_error" {
				t.Fatalf("error code = %v, want %q", chunk.BifrostError.Error.Code, "nested_error")
			}
			return
		case <-timeout:
			t.Fatal("timed out waiting for stream error")
		}
	}
}

func TestResponsesLargePayloadOpenAIToChatCompletionsConvertsBody(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		bodyStr := string(body)
		if strings.Contains(bodyStr, `"input"`) {
			t.Fatalf("request body still contains responses input: %s", bodyStr)
		}
		if !strings.Contains(bodyStr, `"messages"`) {
			t.Fatalf("request body missing chat messages: %s", bodyStr)
		}
		if !strings.Contains(bodyStr, `"response_format"`) {
			t.Fatalf("request body missing response_format: %s", bodyStr)
		}
		if !strings.Contains(bodyStr, `"user": "responses-user"`) {
			t.Fatalf("request body missing user: %s", bodyStr)
		}

		response := &schemas.BifrostChatResponse{
			ID:      "chatcmpl_large_resp",
			Object:  "chat.completion",
			Created: 123,
			Model:   "deepseek-ai/DeepSeek-R1",
			Choices: []schemas.BifrostResponseChoice{
				{
					Index: 0,
					ChatNonStreamResponseChoice: &schemas.ChatNonStreamResponseChoice{
						Message: &schemas.ChatMessage{
							Role: schemas.ChatMessageRoleAssistant,
							Content: &schemas.ChatMessageContent{
								ContentStr: schemas.Ptr("ok"),
							},
						},
					},
					FinishReason: schemas.Ptr(string(schemas.BifrostFinishReasonStop)),
				},
			},
		}
		_ = sonic.ConfigDefault.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	ctx := newTestContext()
	rawPayload := `{"model":"gmi/deepseek-ai/DeepSeek-R1","input":[{"role":"user","content":"hello"}],"text":{"format":{"type":"json_object"}},"user":"responses-user"}`
	ctx.SetValue(schemas.BifrostContextKeyIntegrationType, "openai")
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadMode, true)
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadReader, strings.NewReader(rawPayload))
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadContentLength, len(rawPayload))
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadContentType, "application/json")
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadMetadata, &schemas.LargePayloadMetadata{
		Model: "gmi/deepseek-ai/DeepSeek-R1",
	})

	request := &schemas.BifrostResponsesRequest{
		Provider: schemas.GMI,
		Model:    "deepseek-ai/DeepSeek-R1",
	}

	provider := newTestProvider(t, server.URL)
	if _, err := provider.Responses(ctx, schemas.Key{Value: schemas.EnvVar{Val: "test-key"}}, request); err != nil {
		t.Fatalf("Responses() error = %v", err)
	}
}

func TestResponsesLargePayloadAnthropicToResponsesConvertsBody(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		bodyStr := string(body)
		if strings.Contains(bodyStr, `"messages"`) {
			t.Fatalf("request body still contains anthropic messages: %s", bodyStr)
		}
		if !strings.Contains(bodyStr, `"input"`) {
			t.Fatalf("request body missing responses input: %s", bodyStr)
		}
		if !strings.Contains(bodyStr, `"max_output_tokens": 64`) {
			t.Fatalf("request body missing converted max_output_tokens: %s", bodyStr)
		}
		if !strings.Contains(bodyStr, `"model": "openai/gpt-5.4"`) {
			t.Fatalf("request body missing normalized model: %s", bodyStr)
		}

		response := &schemas.BifrostResponsesResponse{
			ID:        schemas.Ptr("resp_anthropic_large"),
			Object:    "response",
			CreatedAt: 123,
			Model:     "openai/gpt-5.4",
			Output: []schemas.ResponsesMessage{
				{
					ID:   schemas.Ptr("msg_1"),
					Type: schemas.Ptr(schemas.ResponsesMessageTypeMessage),
					Role: schemas.Ptr(schemas.ResponsesInputMessageRoleAssistant),
					Content: &schemas.ResponsesMessageContent{
						ContentBlocks: []schemas.ResponsesMessageContentBlock{
							{
								Type:                              schemas.ResponsesOutputMessageContentTypeText,
								Text:                              schemas.Ptr("ok"),
								ResponsesOutputMessageContentText: &schemas.ResponsesOutputMessageContentText{},
							},
						},
					},
				},
			},
			StopReason: schemas.Ptr("stop"),
		}
		_ = sonic.ConfigDefault.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	ctx := newTestContext()
	rawPayload := `{"model":"gmi/openai/gpt-5.4","max_tokens":64,"messages":[{"role":"user","content":"hello"}]}`
	ctx.SetValue(schemas.BifrostContextKeyIntegrationType, "anthropic")
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadMode, true)
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadReader, strings.NewReader(rawPayload))
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadContentLength, len(rawPayload))
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadContentType, "application/json")
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadMetadata, &schemas.LargePayloadMetadata{
		Model: "gmi/openai/gpt-5.4",
	})

	request := &schemas.BifrostResponsesRequest{
		Provider: schemas.GMI,
		Model:    "openai/gpt-5.4",
	}

	provider := newTestProvider(t, server.URL)
	if _, err := provider.Responses(ctx, schemas.Key{Value: schemas.EnvVar{Val: "test-key"}}, request); err != nil {
		t.Fatalf("Responses() error = %v", err)
	}
}

func TestResponsesLargePayloadGenAIToChatCompletionsConvertsBody(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		bodyStr := string(body)
		if strings.Contains(bodyStr, `"contents"`) {
			t.Fatalf("request body still contains genai contents: %s", bodyStr)
		}
		if !strings.Contains(bodyStr, `"messages"`) {
			t.Fatalf("request body missing chat messages: %s", bodyStr)
		}
		if !strings.Contains(bodyStr, `"model": "deepseek-ai/DeepSeek-R1"`) {
			t.Fatalf("request body missing model: %s", bodyStr)
		}

		response := &schemas.BifrostChatResponse{
			ID:      "chatcmpl_genai_large",
			Object:  "chat.completion",
			Created: 123,
			Model:   "deepseek-ai/DeepSeek-R1",
			Choices: []schemas.BifrostResponseChoice{
				{
					Index: 0,
					ChatNonStreamResponseChoice: &schemas.ChatNonStreamResponseChoice{
						Message: &schemas.ChatMessage{
							Role: schemas.ChatMessageRoleAssistant,
							Content: &schemas.ChatMessageContent{
								ContentStr: schemas.Ptr("ok"),
							},
						},
					},
					FinishReason: schemas.Ptr(string(schemas.BifrostFinishReasonStop)),
				},
			},
		}
		_ = sonic.ConfigDefault.NewEncoder(w).Encode(response)
	}))
	defer server.Close()

	ctx := newTestContext()
	rawPayload := `{"contents":[{"role":"user","parts":[{"text":"hello"}]}]}`
	ctx.SetValue(schemas.BifrostContextKeyIntegrationType, "genai")
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadMode, true)
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadReader, strings.NewReader(rawPayload))
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadContentLength, len(rawPayload))
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadContentType, "application/json")
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadMetadata, &schemas.LargePayloadMetadata{
		Model: "deepseek-ai/DeepSeek-R1",
	})

	request := &schemas.BifrostResponsesRequest{
		Provider: schemas.GMI,
		Model:    "deepseek-ai/DeepSeek-R1",
	}

	provider := newTestProvider(t, server.URL)
	if _, err := provider.Responses(ctx, schemas.Key{Value: schemas.EnvVar{Val: "test-key"}}, request); err != nil {
		t.Fatalf("Responses() error = %v", err)
	}
}

func TestChatCompletionStreamOpenAIResponsesLargePayloadConvertsBody(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/responses" {
			t.Fatalf("unexpected path %q", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		bodyStr := string(body)
		if strings.Contains(bodyStr, `"model":"gmi/openai/gpt-5.4"`) {
			t.Fatalf("request body still contains gmi-prefixed model: %s", bodyStr)
		}
		if !strings.Contains(bodyStr, `"model": "openai/gpt-5.4"`) {
			t.Fatalf("request body missing normalized model: %s", bodyStr)
		}
		if strings.Contains(bodyStr, `"messages"`) {
			t.Fatalf("request body still contains chat messages: %s", bodyStr)
		}
		if !strings.Contains(bodyStr, `"input"`) {
			t.Fatalf("request body missing responses input: %s", bodyStr)
		}

		w.Header().Set("Content-Type", "text/event-stream")
		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatal("response writer does not support flushing")
		}

		events := []string{
			`{"type":"response.created","sequence_number":0,"response":{"id":"resp_lp","created_at":123,"model":"openai/gpt-5.4"}}`,
			`{"type":"response.completed","sequence_number":1,"response":{"id":"resp_lp","created_at":123,"model":"openai/gpt-5.4","stop_reason":"stop","usage":{"input_tokens":2,"output_tokens":3,"total_tokens":5}}}`,
		}

		for _, event := range events {
			fmt.Fprintf(w, "data: %s\n\n", event)
			flusher.Flush()
		}
	}))
	defer server.Close()

	ctx := newTestContext()
	rawPayload := `{"model":"gmi/openai/gpt-5.4","messages":[{"role":"user","content":"hello"}],"stream":true}`
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadMode, true)
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadReader, strings.NewReader(rawPayload))
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadContentLength, len(rawPayload))
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadContentType, "application/json")
	ctx.SetValue(schemas.BifrostContextKeyIntegrationType, "openai")
	ctx.SetValue(schemas.BifrostContextKeyLargePayloadMetadata, &schemas.LargePayloadMetadata{
		Model: "gmi/openai/gpt-5.4",
	})

	provider := newTestProvider(t, server.URL)
	request := &schemas.BifrostChatRequest{
		Provider: schemas.GMI,
		Model:    "openai/gpt-5.4",
	}
	stream, err := provider.ChatCompletionStream(ctx, noOpPostHookRunner, schemas.Key{Value: schemas.EnvVar{Val: "test-key"}}, request)
	if err != nil {
		t.Fatalf("ChatCompletionStream() error = %v", err)
	}

	timeout := time.After(5 * time.Second)
	for {
		select {
		case chunk, ok := <-stream:
			if !ok {
				return
			}
			if chunk.BifrostError != nil {
				t.Fatalf("unexpected stream error: %#v", chunk.BifrostError)
			}
		case <-timeout:
			t.Fatal("timed out waiting for stream")
		}
	}
}

func TestResponsesToChatStreamStateDerivesSequentialToolCallIndexes(t *testing.T) {
	t.Parallel()

	state := newResponsesToChatStreamState()

	messageType := schemas.ResponsesMessageTypeMessage
	assistantRole := schemas.ResponsesInputMessageRoleAssistant
	state.convert(&schemas.BifrostResponsesStreamResponse{
		Type:        schemas.ResponsesStreamResponseTypeOutputItemAdded,
		OutputIndex: schemas.Ptr(0),
		Item: &schemas.ResponsesMessage{
			ID:      schemas.Ptr("msg_1"),
			Type:    &messageType,
			Role:    &assistantRole,
			Content: &schemas.ResponsesMessageContent{ContentBlocks: []schemas.ResponsesMessageContentBlock{}},
		},
	})

	functionCallType := schemas.ResponsesMessageTypeFunctionCall
	firstToolChunks := state.convert(&schemas.BifrostResponsesStreamResponse{
		Type:        schemas.ResponsesStreamResponseTypeOutputItemAdded,
		OutputIndex: schemas.Ptr(1),
		Item: &schemas.ResponsesMessage{
			ID:   schemas.Ptr("item_1"),
			Type: &functionCallType,
			ResponsesToolMessage: &schemas.ResponsesToolMessage{
				CallID: schemas.Ptr("call_1"),
				Name:   schemas.Ptr("tool_one"),
			},
		},
	})
	if len(firstToolChunks) != 1 {
		t.Fatalf("first tool call produced %d chunks, want 1", len(firstToolChunks))
	}
	if got := firstToolChunks[0].Choices[0].ChatStreamResponseChoice.Delta.ToolCalls[0].Index; got != 0 {
		t.Fatalf("first tool call index = %d, want 0", got)
	}

	firstArgsChunks := state.convert(&schemas.BifrostResponsesStreamResponse{
		Type:   schemas.ResponsesStreamResponseTypeFunctionCallArgumentsDelta,
		ItemID: schemas.Ptr("item_1"),
		Delta:  schemas.Ptr(`{"alpha":1}`),
	})
	if len(firstArgsChunks) != 1 {
		t.Fatalf("first tool args produced %d chunks, want 1", len(firstArgsChunks))
	}
	if got := firstArgsChunks[0].Choices[0].ChatStreamResponseChoice.Delta.ToolCalls[0].Index; got != 0 {
		t.Fatalf("first tool args index = %d, want 0", got)
	}

	secondToolChunks := state.convert(&schemas.BifrostResponsesStreamResponse{
		Type:        schemas.ResponsesStreamResponseTypeOutputItemAdded,
		OutputIndex: schemas.Ptr(3),
		Item: &schemas.ResponsesMessage{
			ID:   schemas.Ptr("item_2"),
			Type: &functionCallType,
			ResponsesToolMessage: &schemas.ResponsesToolMessage{
				CallID: schemas.Ptr("call_2"),
				Name:   schemas.Ptr("tool_two"),
			},
		},
	})
	if len(secondToolChunks) != 1 {
		t.Fatalf("second tool call produced %d chunks, want 1", len(secondToolChunks))
	}
	if got := secondToolChunks[0].Choices[0].ChatStreamResponseChoice.Delta.ToolCalls[0].Index; got != 1 {
		t.Fatalf("second tool call index = %d, want 1", got)
	}

	secondArgsChunks := state.convert(&schemas.BifrostResponsesStreamResponse{
		Type:        schemas.ResponsesStreamResponseTypeFunctionCallArgumentsDelta,
		OutputIndex: schemas.Ptr(3),
		Delta:       schemas.Ptr(`{"beta":2}`),
	})
	if len(secondArgsChunks) != 1 {
		t.Fatalf("second tool args produced %d chunks, want 1", len(secondArgsChunks))
	}
	if got := secondArgsChunks[0].Choices[0].ChatStreamResponseChoice.Delta.ToolCalls[0].Index; got != 1 {
		t.Fatalf("second tool args index = %d, want 1", got)
	}
}
