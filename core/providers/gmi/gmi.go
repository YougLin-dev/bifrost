package gmi

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/bytedance/sonic"
	"github.com/valyala/fasthttp"

	"github.com/maximhq/bifrost/core/providers/anthropic"
	"github.com/maximhq/bifrost/core/providers/gemini"
	"github.com/maximhq/bifrost/core/providers/openai"
	providerUtils "github.com/maximhq/bifrost/core/providers/utils"
	schemas "github.com/maximhq/bifrost/core/schemas"
)

const (
	defaultGMIBaseURL       = "https://api.gmi-serving.com"
	defaultAnthropicVersion = "2023-06-01"
)

type modelFamily string

const (
	modelFamilyResponses       modelFamily = "responses"
	modelFamilyChatCompletions modelFamily = "chat_completions"
	modelFamilyAnthropic       modelFamily = "anthropic"
	modelFamilyGoogle          modelFamily = "google"
)

type resolvedModel struct {
	family          modelFamily
	upstreamModel   string
	requestModel    string
	modelDeployment string
}

type GMIProvider struct {
	logger               schemas.Logger
	client               *fasthttp.Client
	networkConfig        schemas.NetworkConfig
	sendBackRawRequest   bool
	sendBackRawResponse  bool
	customProviderConfig *schemas.CustomProviderConfig
}

func NewGMIProvider(config *schemas.ProviderConfig, logger schemas.Logger) (*GMIProvider, error) {
	config.CheckAndSetDefaults()

	requestTimeout := time.Second * time.Duration(config.NetworkConfig.DefaultRequestTimeoutInSeconds)
	client := &fasthttp.Client{
		ReadTimeout:         requestTimeout,
		WriteTimeout:        requestTimeout,
		MaxConnsPerHost:     config.NetworkConfig.MaxConnsPerHost,
		MaxIdleConnDuration: 30 * time.Second,
		MaxConnWaitTimeout:  requestTimeout,
		MaxConnDuration:     time.Second * time.Duration(schemas.DefaultMaxConnDurationInSeconds),
		ConnPoolStrategy:    fasthttp.FIFO,
	}

	client = providerUtils.ConfigureProxy(client, config.ProxyConfig, logger)
	client = providerUtils.ConfigureDialer(client)
	client = providerUtils.ConfigureTLS(client, config.NetworkConfig, logger)

	if config.NetworkConfig.BaseURL == "" {
		config.NetworkConfig.BaseURL = defaultGMIBaseURL
	}
	config.NetworkConfig.BaseURL = strings.TrimRight(config.NetworkConfig.BaseURL, "/")

	return &GMIProvider{
		logger:               logger,
		client:               client,
		networkConfig:        config.NetworkConfig,
		sendBackRawRequest:   config.SendBackRawRequest,
		sendBackRawResponse:  config.SendBackRawResponse,
		customProviderConfig: config.CustomProviderConfig,
	}, nil
}

func (provider *GMIProvider) GetProviderKey() schemas.ModelProvider {
	return providerUtils.GetProviderName(schemas.GMI, provider.customProviderConfig)
}

func isAnthropicNamespace(model string) bool {
	return strings.HasPrefix(model, string(modelFamilyAnthropic)+"/")
}

func isGoogleNamespace(model string) bool {
	return strings.HasPrefix(model, string(modelFamilyGoogle)+"/")
}

func resolveModel(model string, providerKey schemas.ModelProvider) (*resolvedModel, *schemas.BifrostError) {
	familyStr, upstreamModel, ok := strings.Cut(model, "/")
	if !ok || strings.TrimSpace(upstreamModel) == "" {
		return nil, providerUtils.NewConfigurationError("gmi model must be in family/model format", providerKey)
	}

	resolved := &resolvedModel{
		requestModel: model,
	}

	switch modelFamily(strings.TrimSpace(familyStr)) {
	case "openai":
		resolved.family = modelFamilyResponses
		resolved.upstreamModel = model
		resolved.modelDeployment = model
	case modelFamilyAnthropic:
		resolved.family = modelFamilyAnthropic
		resolved.upstreamModel = model
		resolved.modelDeployment = model
	case modelFamilyGoogle:
		resolved.family = modelFamilyGoogle
		resolved.upstreamModel = upstreamModel
		resolved.modelDeployment = upstreamModel
	default:
		// GMI exposes many OpenAI-compatible namespaces beyond `openai/*`
		// such as `deepseek-ai/*`, `Qwen/*`, etc. Those models use the
		// chat-completions upstream, while only the explicit `openai/*`
		// namespace uses the Responses API.
		resolved.family = modelFamilyChatCompletions
		resolved.upstreamModel = model
		resolved.modelDeployment = model
	}

	return resolved, nil
}

func (provider *GMIProvider) buildRequestURL(ctx *schemas.BifrostContext, defaultPath string, requestType schemas.RequestType) string {
	path, isCompleteURL := providerUtils.GetRequestPath(ctx, defaultPath, provider.customProviderConfig, requestType)
	if isCompleteURL {
		return path
	}
	return provider.networkConfig.BaseURL + path
}

func (provider *GMIProvider) authorizationHeaders(key schemas.Key) map[string]string {
	headers := map[string]string{}
	if key.Value.GetValue() != "" {
		headers["Authorization"] = "Bearer " + key.Value.GetValue()
	}
	return headers
}

func setResponseExtras(providerKey schemas.ModelProvider, extras *schemas.BifrostResponseExtraFields, requestType schemas.RequestType, model *resolvedModel, latency int64, headers map[string]string) {
	extras.Provider = providerKey
	extras.ModelRequested = model.requestModel
	extras.RequestType = requestType
	extras.Latency = latency
	extras.ProviderResponseHeaders = headers
	if model.modelDeployment != "" && model.modelDeployment != model.requestModel {
		extras.ModelDeployment = model.modelDeployment
	}
}

func setErrorExtras(providerKey schemas.ModelProvider, err *schemas.BifrostError, requestType schemas.RequestType, model *resolvedModel) {
	if err == nil {
		return
	}
	err.ExtraFields.Provider = providerKey
	err.ExtraFields.ModelRequested = model.requestModel
	err.ExtraFields.RequestType = requestType
}

func (provider *GMIProvider) buildAnthropicHeaders(ctx *schemas.BifrostContext, key schemas.Key) map[string]string {
	headers := provider.authorizationHeaders(key)
	headers["anthropic-version"] = defaultAnthropicVersion
	if betaHeaders := anthropic.FilterBetaHeadersForProvider(
		anthropic.MergeBetaHeaders(provider.networkConfig.ExtraHeaders, ctx),
		provider.GetProviderKey(),
		provider.networkConfig.BetaHeaderOverrides,
	); len(betaHeaders) > 0 {
		headers[anthropic.AnthropicBetaHeader] = strings.Join(betaHeaders, ",")
	}
	return headers
}

func parseAnthropicCompatibleError(resp *fasthttp.Response, meta *providerUtils.RequestMetadata) *schemas.BifrostError {
	var errorResp anthropic.AnthropicError
	bifrostErr := providerUtils.HandleProviderAPIError(resp, &errorResp)
	if errorResp.Error != nil {
		if bifrostErr.Error == nil {
			bifrostErr.Error = &schemas.ErrorField{}
		}
		bifrostErr.Error.Type = &errorResp.Error.Type
		bifrostErr.Error.Message = errorResp.Error.Message
	}
	if meta != nil {
		bifrostErr.ExtraFields.Provider = meta.Provider
		bifrostErr.ExtraFields.ModelRequested = meta.Model
		bifrostErr.ExtraFields.RequestType = meta.RequestType
	}
	return bifrostErr
}

func parseGeminiCompatibleError(resp *fasthttp.Response, meta *providerUtils.RequestMetadata) *schemas.BifrostError {
	var errorResps []gemini.GeminiGenerationError
	bifrostErr := providerUtils.HandleProviderAPIError(resp, &errorResps)
	if len(errorResps) > 0 {
		if bifrostErr.Error == nil {
			bifrostErr.Error = &schemas.ErrorField{}
		}
		var messages []string
		for _, errorResp := range errorResps {
			if errorResp.Error == nil {
				continue
			}
			messages = append(messages, errorResp.Error.Message)
			if bifrostErr.Error.Code == nil {
				bifrostErr.Error.Code = schemas.Ptr(strconv.Itoa(errorResp.Error.Code))
			}
		}
		if len(messages) > 0 {
			bifrostErr.Error.Message = strings.Join(messages, "\n")
		}
	} else {
		var errorResp gemini.GeminiGenerationError
		bifrostErr = providerUtils.HandleProviderAPIError(resp, &errorResp)
		if errorResp.Error != nil {
			if bifrostErr.Error == nil {
				bifrostErr.Error = &schemas.ErrorField{}
			}
			bifrostErr.Error.Code = schemas.Ptr(strconv.Itoa(errorResp.Error.Code))
			bifrostErr.Error.Message = errorResp.Error.Message
		}
	}
	if meta != nil {
		bifrostErr.ExtraFields.Provider = meta.Provider
		bifrostErr.ExtraFields.ModelRequested = meta.Model
		bifrostErr.ExtraFields.RequestType = meta.RequestType
	}
	return bifrostErr
}

type gmiErrorParser func(resp *fasthttp.Response, meta *providerUtils.RequestMetadata) *schemas.BifrostError

func (provider *GMIProvider) doJSONRequest(
	ctx *schemas.BifrostContext,
	url string,
	jsonBody []byte,
	headers map[string]string,
	skipExtraHeaders []string,
	meta *providerUtils.RequestMetadata,
	errorParser gmiErrorParser,
) ([]byte, time.Duration, map[string]string, bool, *schemas.BifrostError) {
	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	respOwned := true
	defer func() {
		if respOwned {
			fasthttp.ReleaseResponse(resp)
		}
	}()

	req.Header.SetMethod(http.MethodPost)
	req.SetRequestURI(url)
	req.Header.SetContentType("application/json")
	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, skipExtraHeaders)
	for key, value := range headers {
		req.Header.Set(key, value)
	}

	usedLargePayloadBody := providerUtils.ApplyLargePayloadRequestBody(ctx, req)
	if !usedLargePayloadBody {
		req.SetBody(jsonBody)
	}

	activeClient := providerUtils.PrepareResponseStreaming(ctx, provider.client, resp)
	latency, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, activeClient, req, resp)
	defer wait()
	if bifrostErr != nil {
		return nil, latency, nil, false, bifrostErr
	}
	if usedLargePayloadBody {
		providerUtils.DrainLargePayloadRemainder(ctx)
	}

	providerHeaders := providerUtils.ExtractProviderResponseHeaders(resp)
	ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, providerHeaders)

	if resp.StatusCode() != fasthttp.StatusOK {
		providerUtils.MaterializeStreamErrorBody(ctx, resp)
		return nil, latency, providerHeaders, false, errorParser(resp, meta)
	}

	body, isLargeResp, decodeErr := providerUtils.FinalizeResponseWithLargeDetection(ctx, resp, provider.GetProviderKey(), provider.logger)
	if decodeErr != nil {
		return nil, latency, providerHeaders, false, decodeErr
	}
	if isLargeResp {
		respOwned = false
		return nil, latency, providerHeaders, true, nil
	}

	return body, latency, providerHeaders, false, nil
}

func normalizeGMIModelList(response *openai.OpenAIListModelsResponse) {
	if response == nil {
		return
	}
	for i := range response.Data {
		if strings.Contains(response.Data[i].ID, "/") {
			continue
		}
		ownedBy := strings.ToLower(strings.TrimSpace(response.Data[i].OwnedBy))
		switch {
		case strings.Contains(ownedBy, "anthropic"):
			response.Data[i].ID = "anthropic/" + response.Data[i].ID
		case strings.Contains(ownedBy, "google"), strings.Contains(ownedBy, "gemini"):
			response.Data[i].ID = "google/" + response.Data[i].ID
		case strings.Contains(ownedBy, "openai"):
			response.Data[i].ID = "openai/" + response.Data[i].ID
		}
	}
}

func (provider *GMIProvider) listModelsInternal(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostListModelsRequest) (*schemas.BifrostListModelsResponse, *schemas.BifrostError) {
	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	defer fasthttp.ReleaseRequest(req)
	defer fasthttp.ReleaseResponse(resp)

	req.Header.SetMethod(http.MethodGet)
	req.SetRequestURI(provider.buildRequestURL(ctx, "/v1/models", schemas.ListModelsRequest))
	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)
	if len(keys) > 0 && keys[0].Value.GetValue() != "" {
		req.Header.Set("Authorization", "Bearer "+keys[0].Value.GetValue())
	}

	latency, bifrostErr, wait := providerUtils.MakeRequestWithContext(ctx, provider.client, req, resp)
	defer wait()
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, providerUtils.ExtractProviderResponseHeaders(resp))

	if resp.StatusCode() != fasthttp.StatusOK {
		return nil, openai.ParseOpenAIError(resp, schemas.ListModelsRequest, provider.GetProviderKey(), "")
	}

	var openAIResp openai.OpenAIListModelsResponse
	rawRequest, rawResponse, bifrostErr := providerUtils.HandleProviderResponse(
		resp.Body(),
		&openAIResp,
		nil,
		providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
		providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
	)
	if bifrostErr != nil {
		return nil, bifrostErr
	}
	normalizeGMIModelList(&openAIResp)

	allowedModels := []string{}
	blacklistedModels := []string{}
	if len(keys) > 0 {
		allowedModels = keys[0].Models
		blacklistedModels = keys[0].BlacklistedModels
	}

	response := openAIResp.ToBifrostListModelsResponse(provider.GetProviderKey(), allowedModels, blacklistedModels, request.Unfiltered)
	response.ExtraFields.Provider = provider.GetProviderKey()
	response.ExtraFields.RequestType = schemas.ListModelsRequest
	response.ExtraFields.Latency = latency.Milliseconds()
	response.ExtraFields.ProviderResponseHeaders = providerUtils.ExtractProviderResponseHeaders(resp)
	if providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest) {
		response.ExtraFields.RawRequest = rawRequest
	}
	if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
		response.ExtraFields.RawResponse = rawResponse
	}

	return response, nil
}

func (provider *GMIProvider) ListModels(ctx *schemas.BifrostContext, keys []schemas.Key, request *schemas.BifrostListModelsRequest) (*schemas.BifrostListModelsResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.GMI, provider.customProviderConfig, schemas.ListModelsRequest); err != nil {
		return nil, err
	}
	if provider.customProviderConfig != nil && provider.customProviderConfig.IsKeyLess {
		return providerUtils.HandleKeylessListModelsRequest(provider.GetProviderKey(), func() (*schemas.BifrostListModelsResponse, *schemas.BifrostError) {
			return provider.listModelsInternal(ctx, nil, request)
		})
	}
	return provider.listModelsInternal(ctx, keys, request)
}

func mapResponsesStopReasonToChat(stopReason *string) *string {
	if stopReason == nil || strings.TrimSpace(*stopReason) == "" {
		return nil
	}
	switch strings.TrimSpace(*stopReason) {
	case "stop", "end_turn", "stop_sequence":
		return schemas.Ptr(string(schemas.BifrostFinishReasonStop))
	case "tool_use", "tool_calls":
		return schemas.Ptr(string(schemas.BifrostFinishReasonToolCalls))
	case "max_tokens", "max_output_tokens", "length":
		return schemas.Ptr(string(schemas.BifrostFinishReasonLength))
	default:
		return schemas.Ptr(*stopReason)
	}
}

func finalizeChatResponseFromResponses(response *schemas.BifrostResponsesResponse) *schemas.BifrostChatResponse {
	if response == nil {
		return nil
	}
	chatResponse := response.ToBifrostChatResponse()
	finishReason := mapResponsesStopReasonToChat(response.StopReason)
	if finishReason != nil && len(chatResponse.Choices) > 0 && chatResponse.Choices[0].FinishReason == nil {
		chatResponse.Choices[0].FinishReason = finishReason
	}
	return chatResponse
}

func (provider *GMIProvider) ChatCompletion(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostChatRequest) (*schemas.BifrostChatResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.GMI, provider.customProviderConfig, schemas.ChatCompletionRequest); err != nil {
		return nil, err
	}
	model, bifrostErr := resolveModel(request.Model, provider.GetProviderKey())
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	switch model.family {
	case modelFamilyResponses:
		responsesReq := request.ToResponsesRequest()
		response, bifrostErr := openai.HandleOpenAIResponsesRequest(
			ctx,
			provider.client,
			provider.buildRequestURL(ctx, "/v1/responses", schemas.ChatCompletionRequest),
			responsesReq,
			key,
			provider.networkConfig.ExtraHeaders,
			providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
			providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
			provider.GetProviderKey(),
			nil,
			nil,
			provider.logger,
		)
		if bifrostErr != nil {
			setErrorExtras(provider.GetProviderKey(), bifrostErr, schemas.ChatCompletionRequest, model)
			return nil, bifrostErr
		}
		chatResponse := finalizeChatResponseFromResponses(response)
		if chatResponse == nil {
			return nil, providerUtils.NewBifrostOperationError(schemas.ErrProviderResponseEmpty, nil, provider.GetProviderKey())
		}
		setResponseExtras(provider.GetProviderKey(), &chatResponse.ExtraFields, schemas.ChatCompletionRequest, model, response.ExtraFields.Latency, response.ExtraFields.ProviderResponseHeaders)
		chatResponse.ExtraFields.RawRequest = response.ExtraFields.RawRequest
		chatResponse.ExtraFields.RawResponse = response.ExtraFields.RawResponse
		return chatResponse, nil

	case modelFamilyChatCompletions:
		response, bifrostErr := openai.HandleOpenAIChatCompletionRequest(
			ctx,
			provider.client,
			provider.buildRequestURL(ctx, "/v1/chat/completions", schemas.ChatCompletionRequest),
			request,
			key,
			provider.networkConfig.ExtraHeaders,
			providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
			providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
			provider.GetProviderKey(),
			nil,
			nil,
			provider.logger,
		)
		if bifrostErr != nil {
			setErrorExtras(provider.GetProviderKey(), bifrostErr, schemas.ChatCompletionRequest, model)
			return nil, bifrostErr
		}
		setResponseExtras(provider.GetProviderKey(), &response.ExtraFields, schemas.ChatCompletionRequest, model, response.ExtraFields.Latency, response.ExtraFields.ProviderResponseHeaders)
		return response, nil

	case modelFamilyAnthropic:
		jsonBody, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
			ctx,
			request,
			func() (providerUtils.RequestBodyWithExtraParams, error) {
				reqBody, err := anthropic.ToAnthropicChatRequest(ctx, request)
				if err != nil {
					return nil, err
				}
				if reqBody == nil {
					return nil, fmt.Errorf("chat completion input is not provided")
				}
				reqBody.Model = model.upstreamModel
				anthropic.AddMissingBetaHeadersToContext(ctx, reqBody, schemas.Anthropic)
				return reqBody, nil
			},
			provider.GetProviderKey(),
		)
		if bifrostErr != nil {
			return nil, bifrostErr
		}

		url := provider.buildRequestURL(ctx, "/v1/messages", schemas.ChatCompletionRequest)
		body, latency, providerHeaders, isLargeResp, bifrostErr := provider.doJSONRequest(
			ctx,
			url,
			jsonBody,
			provider.buildAnthropicHeaders(ctx, key),
			[]string{anthropic.AnthropicBetaHeader},
			&providerUtils.RequestMetadata{
				Provider:    provider.GetProviderKey(),
				Model:       request.Model,
				RequestType: schemas.ChatCompletionRequest,
			},
			parseAnthropicCompatibleError,
		)
		if bifrostErr != nil {
			return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonBody, nil, provider.sendBackRawRequest, provider.sendBackRawResponse)
		}
		if isLargeResp {
			return &schemas.BifrostChatResponse{
				Model: request.Model,
				ExtraFields: schemas.BifrostResponseExtraFields{
					Provider:                provider.GetProviderKey(),
					ModelRequested:          request.Model,
					RequestType:             schemas.ChatCompletionRequest,
					Latency:                 latency.Milliseconds(),
					ModelDeployment:         model.modelDeployment,
					ProviderResponseHeaders: providerHeaders,
				},
			}, nil
		}

		anthropicResponse := anthropic.AcquireAnthropicMessageResponse()
		defer anthropic.ReleaseAnthropicMessageResponse(anthropicResponse)
		rawRequest, rawResponse, bifrostErr := providerUtils.HandleProviderResponse(
			body,
			anthropicResponse,
			jsonBody,
			providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
			providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		)
		if bifrostErr != nil {
			return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonBody, body, provider.sendBackRawRequest, provider.sendBackRawResponse)
		}

		chatResponse := anthropicResponse.ToBifrostChatResponse(ctx)
		setResponseExtras(provider.GetProviderKey(), &chatResponse.ExtraFields, schemas.ChatCompletionRequest, model, latency.Milliseconds(), providerHeaders)
		if providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest) {
			chatResponse.ExtraFields.RawRequest = rawRequest
		}
		if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
			chatResponse.ExtraFields.RawResponse = rawResponse
		}
		return chatResponse, nil

	case modelFamilyGoogle:
		jsonBody, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
			ctx,
			request,
			func() (providerUtils.RequestBodyWithExtraParams, error) {
				reqBody, err := gemini.ToGeminiChatCompletionRequest(request)
				if err != nil {
					return nil, err
				}
				if reqBody == nil {
					return nil, fmt.Errorf("chat completion input is not provided")
				}
				reqBody.Model = model.upstreamModel
				return reqBody, nil
			},
			provider.GetProviderKey(),
		)
		if bifrostErr != nil {
			return nil, bifrostErr
		}

		url := provider.buildRequestURL(
			ctx,
			fmt.Sprintf("/v1/models/%s:generateContent", model.upstreamModel),
			schemas.ChatCompletionRequest,
		)
		body, latency, providerHeaders, isLargeResp, bifrostErr := provider.doJSONRequest(
			ctx,
			url,
			jsonBody,
			provider.authorizationHeaders(key),
			nil,
			&providerUtils.RequestMetadata{
				Provider:    provider.GetProviderKey(),
				Model:       request.Model,
				RequestType: schemas.ChatCompletionRequest,
			},
			parseGeminiCompatibleError,
		)
		if bifrostErr != nil {
			return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonBody, nil, provider.sendBackRawRequest, provider.sendBackRawResponse)
		}
		if isLargeResp {
			return &schemas.BifrostChatResponse{
				Model: request.Model,
				ExtraFields: schemas.BifrostResponseExtraFields{
					Provider:                provider.GetProviderKey(),
					ModelRequested:          request.Model,
					ModelDeployment:         model.modelDeployment,
					RequestType:             schemas.ChatCompletionRequest,
					Latency:                 latency.Milliseconds(),
					ProviderResponseHeaders: providerHeaders,
				},
			}, nil
		}

		geminiResponse := &gemini.GenerateContentResponse{}
		rawRequest, rawResponse, bifrostErr := providerUtils.HandleProviderResponse(
			body,
			geminiResponse,
			jsonBody,
			providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
			providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		)
		if bifrostErr != nil {
			return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonBody, body, provider.sendBackRawRequest, provider.sendBackRawResponse)
		}

		chatResponse := geminiResponse.ToBifrostChatResponse()
		setResponseExtras(provider.GetProviderKey(), &chatResponse.ExtraFields, schemas.ChatCompletionRequest, model, latency.Milliseconds(), providerHeaders)
		if providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest) {
			chatResponse.ExtraFields.RawRequest = rawRequest
		}
		if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
			chatResponse.ExtraFields.RawResponse = rawResponse
		}
		return chatResponse, nil
	}

	return nil, providerUtils.NewConfigurationError("unsupported gmi model family", provider.GetProviderKey())
}

func (provider *GMIProvider) Responses(ctx *schemas.BifrostContext, key schemas.Key, request *schemas.BifrostResponsesRequest) (*schemas.BifrostResponsesResponse, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.GMI, provider.customProviderConfig, schemas.ResponsesRequest); err != nil {
		return nil, err
	}
	model, bifrostErr := resolveModel(request.Model, provider.GetProviderKey())
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	switch model.family {
	case modelFamilyResponses:
		response, bifrostErr := openai.HandleOpenAIResponsesRequest(
			ctx,
			provider.client,
			provider.buildRequestURL(ctx, "/v1/responses", schemas.ResponsesRequest),
			request,
			key,
			provider.networkConfig.ExtraHeaders,
			providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
			providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
			provider.GetProviderKey(),
			nil,
			nil,
			provider.logger,
		)
		if bifrostErr != nil {
			setErrorExtras(provider.GetProviderKey(), bifrostErr, schemas.ResponsesRequest, model)
			return nil, bifrostErr
		}
		setResponseExtras(provider.GetProviderKey(), &response.ExtraFields, schemas.ResponsesRequest, model, response.ExtraFields.Latency, response.ExtraFields.ProviderResponseHeaders)
		return response, nil

	case modelFamilyChatCompletions:
		chatResponse, bifrostErr := openai.HandleOpenAIChatCompletionRequest(
			ctx,
			provider.client,
			provider.buildRequestURL(ctx, "/v1/chat/completions", schemas.ResponsesRequest),
			request.ToChatRequest(),
			key,
			provider.networkConfig.ExtraHeaders,
			providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
			providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
			provider.GetProviderKey(),
			nil,
			nil,
			provider.logger,
		)
		if bifrostErr != nil {
			setErrorExtras(provider.GetProviderKey(), bifrostErr, schemas.ResponsesRequest, model)
			return nil, bifrostErr
		}
		response := chatResponse.ToBifrostResponsesResponse()
		setResponseExtras(provider.GetProviderKey(), &response.ExtraFields, schemas.ResponsesRequest, model, chatResponse.ExtraFields.Latency, chatResponse.ExtraFields.ProviderResponseHeaders)
		return response, nil

	case modelFamilyAnthropic:
		jsonBody, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
			ctx,
			request,
			func() (providerUtils.RequestBodyWithExtraParams, error) {
				reqBody, err := anthropic.ToAnthropicResponsesRequest(ctx, request)
				if err != nil {
					return nil, err
				}
				if reqBody == nil {
					return nil, fmt.Errorf("responses input is not provided")
				}
				reqBody.Model = model.upstreamModel
				anthropic.AddMissingBetaHeadersToContext(ctx, reqBody, schemas.Anthropic)
				return reqBody, nil
			},
			provider.GetProviderKey(),
		)
		if bifrostErr != nil {
			return nil, bifrostErr
		}

		body, latency, providerHeaders, isLargeResp, bifrostErr := provider.doJSONRequest(
			ctx,
			provider.buildRequestURL(ctx, "/v1/messages", schemas.ResponsesRequest),
			jsonBody,
			provider.buildAnthropicHeaders(ctx, key),
			[]string{anthropic.AnthropicBetaHeader},
			&providerUtils.RequestMetadata{
				Provider:    provider.GetProviderKey(),
				Model:       request.Model,
				RequestType: schemas.ResponsesRequest,
			},
			parseAnthropicCompatibleError,
		)
		if bifrostErr != nil {
			return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonBody, nil, provider.sendBackRawRequest, provider.sendBackRawResponse)
		}
		if isLargeResp {
			return &schemas.BifrostResponsesResponse{
				Model: request.Model,
				ExtraFields: schemas.BifrostResponseExtraFields{
					Provider:                provider.GetProviderKey(),
					ModelRequested:          request.Model,
					ModelDeployment:         model.modelDeployment,
					RequestType:             schemas.ResponsesRequest,
					Latency:                 latency.Milliseconds(),
					ProviderResponseHeaders: providerHeaders,
				},
			}, nil
		}

		anthropicResponse := anthropic.AcquireAnthropicMessageResponse()
		defer anthropic.ReleaseAnthropicMessageResponse(anthropicResponse)
		rawRequest, rawResponse, bifrostErr := providerUtils.HandleProviderResponse(
			body,
			anthropicResponse,
			jsonBody,
			providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
			providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		)
		if bifrostErr != nil {
			return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonBody, body, provider.sendBackRawRequest, provider.sendBackRawResponse)
		}

		response := anthropicResponse.ToBifrostResponsesResponse(ctx)
		setResponseExtras(provider.GetProviderKey(), &response.ExtraFields, schemas.ResponsesRequest, model, latency.Milliseconds(), providerHeaders)
		if providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest) {
			response.ExtraFields.RawRequest = rawRequest
		}
		if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
			response.ExtraFields.RawResponse = rawResponse
		}
		return response, nil

	case modelFamilyGoogle:
		jsonBody, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
			ctx,
			request,
			func() (providerUtils.RequestBodyWithExtraParams, error) {
				reqBody, err := gemini.ToGeminiResponsesRequest(request)
				if err != nil {
					return nil, err
				}
				if reqBody == nil {
					return nil, fmt.Errorf("responses input is not provided")
				}
				reqBody.Model = model.upstreamModel
				return reqBody, nil
			},
			provider.GetProviderKey(),
		)
		if bifrostErr != nil {
			return nil, bifrostErr
		}

		body, latency, providerHeaders, isLargeResp, bifrostErr := provider.doJSONRequest(
			ctx,
			provider.buildRequestURL(ctx, fmt.Sprintf("/v1/models/%s:generateContent", model.upstreamModel), schemas.ResponsesRequest),
			jsonBody,
			provider.authorizationHeaders(key),
			nil,
			&providerUtils.RequestMetadata{
				Provider:    provider.GetProviderKey(),
				Model:       request.Model,
				RequestType: schemas.ResponsesRequest,
			},
			parseGeminiCompatibleError,
		)
		if bifrostErr != nil {
			return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonBody, nil, provider.sendBackRawRequest, provider.sendBackRawResponse)
		}
		if isLargeResp {
			return &schemas.BifrostResponsesResponse{
				Model: request.Model,
				ExtraFields: schemas.BifrostResponseExtraFields{
					Provider:                provider.GetProviderKey(),
					ModelRequested:          request.Model,
					ModelDeployment:         model.modelDeployment,
					RequestType:             schemas.ResponsesRequest,
					Latency:                 latency.Milliseconds(),
					ProviderResponseHeaders: providerHeaders,
				},
			}, nil
		}

		geminiResponse := &gemini.GenerateContentResponse{}
		rawRequest, rawResponse, bifrostErr := providerUtils.HandleProviderResponse(
			body,
			geminiResponse,
			jsonBody,
			providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
			providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
		)
		if bifrostErr != nil {
			return nil, providerUtils.EnrichError(ctx, bifrostErr, jsonBody, body, provider.sendBackRawRequest, provider.sendBackRawResponse)
		}

		response := geminiResponse.ToResponsesBifrostResponsesResponse()
		setResponseExtras(provider.GetProviderKey(), &response.ExtraFields, schemas.ResponsesRequest, model, latency.Milliseconds(), providerHeaders)
		if providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest) {
			response.ExtraFields.RawRequest = rawRequest
		}
		if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
			response.ExtraFields.RawResponse = rawResponse
		}
		return response, nil
	}

	return nil, providerUtils.NewConfigurationError("unsupported gmi model family", provider.GetProviderKey())
}

type responsesToChatStreamState struct {
	messageID         string
	model             string
	roleEmitted       bool
	chunkIndex        int
	nextToolCallIndex uint16
	outputIndexes     map[int]uint16
	itemIndexes       map[string]uint16
	callIndexes       map[string]uint16
	terminalUsage     *schemas.BifrostLLMUsage
	terminalReason    *string
	terminalReceived  bool
}

func newResponsesToChatStreamState() *responsesToChatStreamState {
	return &responsesToChatStreamState{
		chunkIndex:    -1,
		outputIndexes: make(map[int]uint16),
		itemIndexes:   make(map[string]uint16),
		callIndexes:   make(map[string]uint16),
	}
}

func (state *responsesToChatStreamState) remember(resp *schemas.BifrostResponsesStreamResponse) {
	if resp == nil {
		return
	}
	if state.messageID == "" {
		switch {
		case resp.Response != nil && resp.Response.ID != nil && *resp.Response.ID != "":
			state.messageID = *resp.Response.ID
		case resp.Item != nil && resp.Item.ID != nil && *resp.Item.ID != "":
			state.messageID = *resp.Item.ID
		}
	}
	if state.model == "" && resp.Response != nil && resp.Response.Model != "" {
		state.model = resp.Response.Model
	}
}

func (state *responsesToChatStreamState) rememberToolCallIndex(resp *schemas.BifrostResponsesStreamResponse, idx uint16) {
	if resp == nil {
		return
	}
	if resp.OutputIndex != nil {
		state.outputIndexes[*resp.OutputIndex] = idx
	}
	if resp.ItemID != nil && *resp.ItemID != "" {
		state.itemIndexes[*resp.ItemID] = idx
	}
	if resp.Item != nil {
		if resp.Item.ID != nil && *resp.Item.ID != "" {
			state.itemIndexes[*resp.Item.ID] = idx
		}
		if resp.Item.CallID != nil && *resp.Item.CallID != "" {
			state.callIndexes[*resp.Item.CallID] = idx
		}
	}
}

func (state *responsesToChatStreamState) resolveToolCallIndex(resp *schemas.BifrostResponsesStreamResponse) uint16 {
	if resp == nil {
		idx := state.nextToolCallIndex
		state.nextToolCallIndex++
		return idx
	}
	if resp.ItemID != nil && *resp.ItemID != "" {
		if idx, ok := state.itemIndexes[*resp.ItemID]; ok {
			return idx
		}
	}
	if resp.OutputIndex != nil {
		if idx, ok := state.outputIndexes[*resp.OutputIndex]; ok {
			return idx
		}
	}
	if resp.Item != nil {
		if resp.Item.ID != nil && *resp.Item.ID != "" {
			if idx, ok := state.itemIndexes[*resp.Item.ID]; ok {
				return idx
			}
		}
		if resp.Item.CallID != nil && *resp.Item.CallID != "" {
			if idx, ok := state.callIndexes[*resp.Item.CallID]; ok {
				return idx
			}
		}
	}
	idx := state.nextToolCallIndex
	state.nextToolCallIndex++
	state.rememberToolCallIndex(resp, idx)
	return idx
}

func (state *responsesToChatStreamState) makeChunk(delta *schemas.ChatStreamResponseChoiceDelta) *schemas.BifrostChatResponse {
	if delta == nil {
		return nil
	}
	state.chunkIndex++
	return &schemas.BifrostChatResponse{
		ID:     state.messageID,
		Model:  state.model,
		Object: "chat.completion.chunk",
		Choices: []schemas.BifrostResponseChoice{
			{
				Index: 0,
				ChatStreamResponseChoice: &schemas.ChatStreamResponseChoice{
					Delta: delta,
				},
			},
		},
	}
}

func (state *responsesToChatStreamState) convert(resp *schemas.BifrostResponsesStreamResponse) []*schemas.BifrostChatResponse {
	if resp == nil {
		return nil
	}
	state.remember(resp)

	var results []*schemas.BifrostChatResponse
	switch resp.Type {
	case schemas.ResponsesStreamResponseTypeOutputItemAdded:
		if resp.Item == nil {
			return nil
		}

		if !state.roleEmitted {
			role := string(schemas.ChatMessageRoleAssistant)
			results = append(results, state.makeChunk(&schemas.ChatStreamResponseChoiceDelta{
				Role: &role,
			}))
			state.roleEmitted = true
		}

		if resp.Item.Type != nil && *resp.Item.Type == schemas.ResponsesMessageTypeFunctionCall && resp.Item.ResponsesToolMessage != nil {
			idx := state.resolveToolCallIndex(resp)
			functionType := string(schemas.ChatToolTypeFunction)
			results = append(results, state.makeChunk(&schemas.ChatStreamResponseChoiceDelta{
				ToolCalls: []schemas.ChatAssistantMessageToolCall{
					{
						Index: idx,
						Type:  &functionType,
						ID:    resp.Item.CallID,
						Function: schemas.ChatAssistantMessageToolCallFunction{
							Name:      resp.Item.Name,
							Arguments: "",
						},
					},
				},
			}))
		}

	case schemas.ResponsesStreamResponseTypeOutputTextDelta:
		if resp.Delta != nil {
			results = append(results, state.makeChunk(&schemas.ChatStreamResponseChoiceDelta{
				Content: resp.Delta,
			}))
		}

	case schemas.ResponsesStreamResponseTypeReasoningSummaryTextDelta:
		if resp.Delta != nil {
			results = append(results, state.makeChunk(&schemas.ChatStreamResponseChoiceDelta{
				Reasoning: resp.Delta,
				ReasoningDetails: []schemas.ChatReasoningDetails{
					{
						Index: state.chunkIndex + 1,
						Type:  schemas.BifrostReasoningDetailsTypeText,
						Text:  resp.Delta,
					},
				},
			}))
		}

	case schemas.ResponsesStreamResponseTypeRefusalDelta:
		if resp.Refusal != nil {
			results = append(results, state.makeChunk(&schemas.ChatStreamResponseChoiceDelta{
				Refusal: resp.Refusal,
			}))
		}

	case schemas.ResponsesStreamResponseTypeFunctionCallArgumentsDelta:
		if resp.Delta != nil {
			idx := state.resolveToolCallIndex(resp)
			results = append(results, state.makeChunk(&schemas.ChatStreamResponseChoiceDelta{
				ToolCalls: []schemas.ChatAssistantMessageToolCall{
					{
						Index: idx,
						Function: schemas.ChatAssistantMessageToolCallFunction{
							Arguments: *resp.Delta,
						},
					},
				},
			}))
		}

	case schemas.ResponsesStreamResponseTypeCompleted, schemas.ResponsesStreamResponseTypeIncomplete:
		state.terminalReceived = true
		if resp.Response != nil {
			if resp.Response.ID != nil && state.messageID == "" {
				state.messageID = *resp.Response.ID
			}
			if resp.Response.Model != "" && state.model == "" {
				state.model = resp.Response.Model
			}
			if resp.Response.Usage != nil {
				state.terminalUsage = resp.Response.Usage.ToBifrostLLMUsage()
			}
			state.terminalReason = mapResponsesStopReasonToChat(resp.Response.StopReason)
			if state.terminalReason == nil && resp.Type == schemas.ResponsesStreamResponseTypeIncomplete {
				state.terminalReason = schemas.Ptr(string(schemas.BifrostFinishReasonLength))
			}
		}
	}

	filtered := results[:0]
	for _, result := range results {
		if result != nil {
			filtered = append(filtered, result)
		}
	}
	return filtered
}

func (provider *GMIProvider) handleOpenAIChatCompletionStreaming(
	ctx *schemas.BifrostContext,
	postHookRunner schemas.PostHookRunner,
	key schemas.Key,
	request *schemas.BifrostChatRequest,
	model *resolvedModel,
) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	responsesReq := request.ToResponsesRequest()
	jsonBody, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
		ctx,
		responsesReq,
		func() (providerUtils.RequestBodyWithExtraParams, error) {
			reqBody := openai.ToOpenAIResponsesRequest(responsesReq)
			if reqBody != nil {
				reqBody.Stream = schemas.Ptr(true)
			}
			return reqBody, nil
		},
		provider.GetProviderKey(),
	)
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	req := fasthttp.AcquireRequest()
	resp := fasthttp.AcquireResponse()
	resp.StreamBody = true
	defer fasthttp.ReleaseRequest(req)

	req.Header.SetMethod(http.MethodPost)
	req.SetRequestURI(provider.buildRequestURL(ctx, "/v1/responses", schemas.ChatCompletionStreamRequest))
	req.Header.SetContentType("application/json")
	providerUtils.SetExtraHeaders(ctx, req, provider.networkConfig.ExtraHeaders, nil)
	if key.Value.GetValue() != "" {
		req.Header.Set("Authorization", "Bearer "+key.Value.GetValue())
	}
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Cache-Control", "no-cache")

	if !providerUtils.ApplyLargePayloadRequestBodyWithModelNormalization(ctx, req, provider.GetProviderKey()) {
		req.SetBody(jsonBody)
	}

	activeClient := providerUtils.PrepareResponseStreaming(ctx, provider.client, resp)
	if err := activeClient.Do(req, resp); err != nil {
		defer providerUtils.ReleaseStreamingResponse(resp)
		if err == context.Canceled {
			return nil, providerUtils.NewBifrostOperationError(schemas.ErrRequestCancelled, err, provider.GetProviderKey())
		}
		if err == context.DeadlineExceeded || err == fasthttp.ErrTimeout {
			return nil, providerUtils.NewBifrostTimeoutError(schemas.ErrProviderRequestTimedOut, err, provider.GetProviderKey())
		}
		return nil, providerUtils.NewBifrostOperationError(schemas.ErrProviderDoRequest, err, provider.GetProviderKey())
	}

	ctx.SetValue(schemas.BifrostContextKeyProviderResponseHeaders, providerUtils.ExtractProviderResponseHeaders(resp))
	if resp.StatusCode() != fasthttp.StatusOK {
		defer providerUtils.ReleaseStreamingResponse(resp)
		providerUtils.MaterializeStreamErrorBody(ctx, resp)
		return nil, providerUtils.EnrichError(
			ctx,
			openai.ParseOpenAIError(resp, schemas.ChatCompletionStreamRequest, provider.GetProviderKey(), request.Model),
			jsonBody,
			nil,
			provider.sendBackRawRequest,
			provider.sendBackRawResponse,
		)
	}

	responseChan := make(chan *schemas.BifrostStreamChunk, schemas.DefaultStreamBufferSize)

	go func() {
		defer func() {
			if ctx.Err() == context.Canceled {
				providerUtils.HandleStreamCancellation(ctx, postHookRunner, responseChan, provider.GetProviderKey(), request.Model, schemas.ChatCompletionStreamRequest, provider.logger)
			} else if ctx.Err() == context.DeadlineExceeded {
				providerUtils.HandleStreamTimeout(ctx, postHookRunner, responseChan, provider.GetProviderKey(), request.Model, schemas.ChatCompletionStreamRequest, provider.logger)
			}
			close(responseChan)
		}()
		defer providerUtils.ReleaseStreamingResponse(resp)

		reader, releaseGzip := providerUtils.DecompressStreamBody(resp)
		defer releaseGzip()
		reader, stopIdleTimeout := providerUtils.NewIdleTimeoutReader(reader, resp.BodyStream(), providerUtils.GetStreamIdleTimeout(ctx))
		defer stopIdleTimeout()
		stopCancellation := providerUtils.SetupStreamCancellation(ctx, resp.BodyStream(), provider.logger)
		defer stopCancellation()

		if providerUtils.DrainNonSSEStreamResponse(resp) {
			ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
			providerUtils.ProcessAndSendError(ctx, postHookRunner, fmt.Errorf("provider returned non-SSE response for streaming request"), responseChan, schemas.ChatCompletionStreamRequest, provider.GetProviderKey(), request.Model, provider.logger)
			return
		}

		sseReader := providerUtils.GetSSEDataReader(ctx, reader)
		state := newResponsesToChatStreamState()
		startTime := time.Now()
		lastChunkTime := startTime

		for {
			if ctx.Err() != nil {
				return
			}

			data, readErr := sseReader.ReadDataLine()
			if readErr != nil {
				if readErr != io.EOF {
					if ctx.Err() != nil {
						return
					}
					ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
					providerUtils.ProcessAndSendError(ctx, postHookRunner, readErr, responseChan, schemas.ChatCompletionStreamRequest, provider.GetProviderKey(), request.Model, provider.logger)
				}
				break
			}

			jsonData := string(data)
			var response schemas.BifrostResponsesStreamResponse
			if err := sonic.UnmarshalString(jsonData, &response); err != nil {
				provider.logger.Warn("failed to parse gmi/openai responses stream chunk: %v", err)
				continue
			}
			if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
				response.ExtraFields.RawResponse = jsonData
			}

			if response.Type == schemas.ResponsesStreamResponseTypeError || response.Type == schemas.ResponsesStreamResponseTypeFailed {
				bifrostErr := &schemas.BifrostError{
					IsBifrostError: false,
					Error:          &schemas.ErrorField{},
					ExtraFields: schemas.BifrostErrorExtraFields{
						Provider:       provider.GetProviderKey(),
						ModelRequested: request.Model,
						RequestType:    schemas.ChatCompletionStreamRequest,
					},
				}
				if response.Message != nil {
					bifrostErr.Error.Message = *response.Message
				}
				if response.Code != nil {
					bifrostErr.Error.Code = response.Code
				}
				ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
				providerUtils.ProcessAndSendBifrostError(
					ctx,
					postHookRunner,
					providerUtils.EnrichError(ctx, bifrostErr, jsonBody, nil, provider.sendBackRawRequest, provider.sendBackRawResponse),
					responseChan,
					provider.logger,
				)
				return
			}

			for _, chatResponse := range state.convert(&response) {
				setResponseExtras(provider.GetProviderKey(), &chatResponse.ExtraFields, schemas.ChatCompletionStreamRequest, model, time.Since(lastChunkTime).Milliseconds(), providerUtils.ExtractProviderResponseHeaders(resp))
				chatResponse.ExtraFields.ChunkIndex = state.chunkIndex
				if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
					chatResponse.ExtraFields.RawResponse = jsonData
				}
				lastChunkTime = time.Now()
				providerUtils.ProcessAndSendResponse(
					ctx,
					postHookRunner,
					providerUtils.GetBifrostResponseForStreamResponse(nil, chatResponse, nil, nil, nil, nil),
					responseChan,
				)
			}

			if state.terminalReceived {
				finalResponse := &schemas.BifrostChatResponse{
					ID:     state.messageID,
					Model:  state.model,
					Object: "chat.completion.chunk",
					Usage:  state.terminalUsage,
					Choices: []schemas.BifrostResponseChoice{
						{
							Index:        0,
							FinishReason: state.terminalReason,
							ChatStreamResponseChoice: &schemas.ChatStreamResponseChoice{
								Delta: &schemas.ChatStreamResponseChoiceDelta{},
							},
						},
					},
				}
				setResponseExtras(provider.GetProviderKey(), &finalResponse.ExtraFields, schemas.ChatCompletionStreamRequest, model, time.Since(startTime).Milliseconds(), providerUtils.ExtractProviderResponseHeaders(resp))
				finalResponse.ExtraFields.ChunkIndex = state.chunkIndex + 1
				if providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest) {
					providerUtils.ParseAndSetRawRequest(&finalResponse.ExtraFields, jsonBody)
				}
				if providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse) {
					finalResponse.ExtraFields.RawResponse = jsonData
				}
				ctx.SetValue(schemas.BifrostContextKeyStreamEndIndicator, true)
				providerUtils.ProcessAndSendResponse(
					ctx,
					postHookRunner,
					providerUtils.GetBifrostResponseForStreamResponse(nil, finalResponse, nil, nil, nil, nil),
					responseChan,
				)
				return
			}
		}
	}()

	return responseChan, nil
}

func (provider *GMIProvider) ChatCompletionStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, key schemas.Key, request *schemas.BifrostChatRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.GMI, provider.customProviderConfig, schemas.ChatCompletionStreamRequest); err != nil {
		return nil, err
	}
	model, bifrostErr := resolveModel(request.Model, provider.GetProviderKey())
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	switch model.family {
	case modelFamilyResponses:
		return provider.handleOpenAIChatCompletionStreaming(ctx, postHookRunner, key, request, model)

	case modelFamilyChatCompletions:
		return openai.HandleOpenAIChatCompletionStreaming(
			ctx,
			provider.client,
			provider.buildRequestURL(ctx, "/v1/chat/completions", schemas.ChatCompletionStreamRequest),
			request,
			provider.authorizationHeaders(key),
			provider.networkConfig.ExtraHeaders,
			providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
			providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
			provider.GetProviderKey(),
			postHookRunner,
			nil,
			nil,
			nil,
			nil,
			nil,
			provider.logger,
		)

	case modelFamilyAnthropic:
		jsonBody, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
			ctx,
			request,
			func() (providerUtils.RequestBodyWithExtraParams, error) {
				reqBody, err := anthropic.ToAnthropicChatRequest(ctx, request)
				if err != nil {
					return nil, err
				}
				if reqBody == nil {
					return nil, fmt.Errorf("chat completion input is not provided")
				}
				reqBody.Model = model.upstreamModel
				reqBody.Stream = schemas.Ptr(true)
				anthropic.AddMissingBetaHeadersToContext(ctx, reqBody, schemas.Anthropic)
				return reqBody, nil
			},
			provider.GetProviderKey(),
		)
		if bifrostErr != nil {
			return nil, bifrostErr
		}
		postResponseConverter := func(response *schemas.BifrostChatResponse) *schemas.BifrostChatResponse {
			setResponseExtras(provider.GetProviderKey(), &response.ExtraFields, schemas.ChatCompletionStreamRequest, model, response.ExtraFields.Latency, response.ExtraFields.ProviderResponseHeaders)
			return response
		}
		return anthropic.HandleAnthropicChatCompletionStreaming(
			ctx,
			provider.client,
			provider.buildRequestURL(ctx, "/v1/messages", schemas.ChatCompletionStreamRequest),
			jsonBody,
			provider.buildAnthropicHeaders(ctx, key),
			provider.networkConfig.ExtraHeaders,
			provider.networkConfig.BetaHeaderOverrides,
			providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
			providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
			provider.GetProviderKey(),
			postHookRunner,
			postResponseConverter,
			provider.logger,
			&providerUtils.RequestMetadata{
				Provider:    provider.GetProviderKey(),
				Model:       request.Model,
				RequestType: schemas.ChatCompletionStreamRequest,
			},
		)

	case modelFamilyGoogle:
		jsonBody, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
			ctx,
			request,
			func() (providerUtils.RequestBodyWithExtraParams, error) {
				reqBody, err := gemini.ToGeminiChatCompletionRequest(request)
				if err != nil {
					return nil, err
				}
				if reqBody == nil {
					return nil, fmt.Errorf("chat completion input is not provided")
				}
				reqBody.Model = model.upstreamModel
				return reqBody, nil
			},
			provider.GetProviderKey(),
		)
		if bifrostErr != nil {
			return nil, bifrostErr
		}
		postResponseConverter := func(response *schemas.BifrostChatResponse) *schemas.BifrostChatResponse {
			setResponseExtras(provider.GetProviderKey(), &response.ExtraFields, schemas.ChatCompletionStreamRequest, model, response.ExtraFields.Latency, response.ExtraFields.ProviderResponseHeaders)
			return response
		}
		return gemini.HandleGeminiChatCompletionStream(
			ctx,
			provider.client,
			provider.buildRequestURL(ctx, fmt.Sprintf("/v1/models/%s:streamGenerateContent", model.upstreamModel), schemas.ChatCompletionStreamRequest),
			jsonBody,
			func() map[string]string {
				headers := provider.authorizationHeaders(key)
				headers["Accept"] = "text/event-stream"
				headers["Cache-Control"] = "no-cache"
				return headers
			}(),
			provider.networkConfig.ExtraHeaders,
			providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
			providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
			provider.GetProviderKey(),
			request.Model,
			postHookRunner,
			postResponseConverter,
			provider.logger,
		)
	}

	return nil, providerUtils.NewConfigurationError("unsupported gmi model family", provider.GetProviderKey())
}

func (provider *GMIProvider) ResponsesStream(ctx *schemas.BifrostContext, postHookRunner schemas.PostHookRunner, key schemas.Key, request *schemas.BifrostResponsesRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	if err := providerUtils.CheckOperationAllowed(schemas.GMI, provider.customProviderConfig, schemas.ResponsesStreamRequest); err != nil {
		return nil, err
	}
	model, bifrostErr := resolveModel(request.Model, provider.GetProviderKey())
	if bifrostErr != nil {
		return nil, bifrostErr
	}

	switch model.family {
	case modelFamilyResponses:
		postResponseConverter := func(response *schemas.BifrostResponsesStreamResponse) *schemas.BifrostResponsesStreamResponse {
			setResponseExtras(provider.GetProviderKey(), &response.ExtraFields, schemas.ResponsesStreamRequest, model, response.ExtraFields.Latency, response.ExtraFields.ProviderResponseHeaders)
			return response
		}
		return openai.HandleOpenAIResponsesStreaming(
			ctx,
			provider.client,
			provider.buildRequestURL(ctx, "/v1/responses", schemas.ResponsesStreamRequest),
			request,
			provider.authorizationHeaders(key),
			provider.networkConfig.ExtraHeaders,
			providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
			providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
			provider.GetProviderKey(),
			postHookRunner,
			nil,
			nil,
			nil,
			postResponseConverter,
			provider.logger,
		)

	case modelFamilyChatCompletions:
		ctx.SetValue(schemas.BifrostContextKeyIsResponsesToChatCompletionFallback, true)
		return openai.HandleOpenAIChatCompletionStreaming(
			ctx,
			provider.client,
			provider.buildRequestURL(ctx, "/v1/chat/completions", schemas.ResponsesStreamRequest),
			request.ToChatRequest(),
			provider.authorizationHeaders(key),
			provider.networkConfig.ExtraHeaders,
			providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
			providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
			provider.GetProviderKey(),
			postHookRunner,
			nil,
			nil,
			nil,
			nil,
			nil,
			provider.logger,
		)

	case modelFamilyAnthropic:
		jsonBody, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
			ctx,
			request,
			func() (providerUtils.RequestBodyWithExtraParams, error) {
				reqBody, err := anthropic.ToAnthropicResponsesRequest(ctx, request)
				if err != nil {
					return nil, err
				}
				if reqBody == nil {
					return nil, fmt.Errorf("responses input is not provided")
				}
				reqBody.Model = model.upstreamModel
				reqBody.Stream = schemas.Ptr(true)
				anthropic.AddMissingBetaHeadersToContext(ctx, reqBody, schemas.Anthropic)
				return reqBody, nil
			},
			provider.GetProviderKey(),
		)
		if bifrostErr != nil {
			return nil, bifrostErr
		}
		postResponseConverter := func(response *schemas.BifrostResponsesStreamResponse) *schemas.BifrostResponsesStreamResponse {
			setResponseExtras(provider.GetProviderKey(), &response.ExtraFields, schemas.ResponsesStreamRequest, model, response.ExtraFields.Latency, response.ExtraFields.ProviderResponseHeaders)
			return response
		}
		return anthropic.HandleAnthropicResponsesStream(
			ctx,
			provider.client,
			provider.buildRequestURL(ctx, "/v1/messages", schemas.ResponsesStreamRequest),
			jsonBody,
			provider.buildAnthropicHeaders(ctx, key),
			provider.networkConfig.ExtraHeaders,
			provider.networkConfig.BetaHeaderOverrides,
			providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
			providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
			provider.GetProviderKey(),
			postHookRunner,
			postResponseConverter,
			provider.logger,
			&providerUtils.RequestMetadata{
				Provider:    provider.GetProviderKey(),
				Model:       request.Model,
				RequestType: schemas.ResponsesStreamRequest,
			},
		)

	case modelFamilyGoogle:
		jsonBody, bifrostErr := providerUtils.CheckContextAndGetRequestBody(
			ctx,
			request,
			func() (providerUtils.RequestBodyWithExtraParams, error) {
				reqBody, err := gemini.ToGeminiResponsesRequest(request)
				if err != nil {
					return nil, err
				}
				if reqBody == nil {
					return nil, fmt.Errorf("responses input is not provided")
				}
				reqBody.Model = model.upstreamModel
				return reqBody, nil
			},
			provider.GetProviderKey(),
		)
		if bifrostErr != nil {
			return nil, bifrostErr
		}
		postResponseConverter := func(response *schemas.BifrostResponsesStreamResponse) *schemas.BifrostResponsesStreamResponse {
			setResponseExtras(provider.GetProviderKey(), &response.ExtraFields, schemas.ResponsesStreamRequest, model, response.ExtraFields.Latency, response.ExtraFields.ProviderResponseHeaders)
			return response
		}
		headers := provider.authorizationHeaders(key)
		headers["Accept"] = "text/event-stream"
		headers["Cache-Control"] = "no-cache"
		return gemini.HandleGeminiResponsesStream(
			ctx,
			provider.client,
			provider.buildRequestURL(ctx, fmt.Sprintf("/v1/models/%s:streamGenerateContent", model.upstreamModel), schemas.ResponsesStreamRequest),
			jsonBody,
			headers,
			provider.networkConfig.ExtraHeaders,
			providerUtils.ShouldSendBackRawRequest(ctx, provider.sendBackRawRequest),
			providerUtils.ShouldSendBackRawResponse(ctx, provider.sendBackRawResponse),
			provider.GetProviderKey(),
			request.Model,
			postHookRunner,
			postResponseConverter,
			provider.logger,
		)
	}

	return nil, providerUtils.NewConfigurationError("unsupported gmi model family", provider.GetProviderKey())
}

func (provider *GMIProvider) TextCompletion(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostTextCompletionRequest) (*schemas.BifrostTextCompletionResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.TextCompletionRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) TextCompletionStream(_ *schemas.BifrostContext, _ schemas.PostHookRunner, _ schemas.Key, _ *schemas.BifrostTextCompletionRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.TextCompletionStreamRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) CountTokens(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostResponsesRequest) (*schemas.BifrostCountTokensResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.CountTokensRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) Embedding(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostEmbeddingRequest) (*schemas.BifrostEmbeddingResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.EmbeddingRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) Rerank(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostRerankRequest) (*schemas.BifrostRerankResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.RerankRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) OCR(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostOCRRequest) (*schemas.BifrostOCRResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.OCRRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) Speech(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostSpeechRequest) (*schemas.BifrostSpeechResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.SpeechRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) SpeechStream(_ *schemas.BifrostContext, _ schemas.PostHookRunner, _ schemas.Key, _ *schemas.BifrostSpeechRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.SpeechStreamRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) Transcription(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostTranscriptionRequest) (*schemas.BifrostTranscriptionResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.TranscriptionRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) TranscriptionStream(_ *schemas.BifrostContext, _ schemas.PostHookRunner, _ schemas.Key, _ *schemas.BifrostTranscriptionRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.TranscriptionStreamRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) ImageGeneration(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostImageGenerationRequest) (*schemas.BifrostImageGenerationResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ImageGenerationRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) ImageGenerationStream(_ *schemas.BifrostContext, _ schemas.PostHookRunner, _ schemas.Key, _ *schemas.BifrostImageGenerationRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ImageGenerationStreamRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) ImageEdit(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostImageEditRequest) (*schemas.BifrostImageGenerationResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ImageEditRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) ImageEditStream(_ *schemas.BifrostContext, _ schemas.PostHookRunner, _ schemas.Key, _ *schemas.BifrostImageEditRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ImageEditStreamRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) ImageVariation(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostImageVariationRequest) (*schemas.BifrostImageGenerationResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ImageVariationRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) VideoGeneration(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostVideoGenerationRequest) (*schemas.BifrostVideoGenerationResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.VideoGenerationRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) VideoRetrieve(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostVideoRetrieveRequest) (*schemas.BifrostVideoGenerationResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.VideoRetrieveRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) VideoDownload(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostVideoDownloadRequest) (*schemas.BifrostVideoDownloadResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.VideoDownloadRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) VideoDelete(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostVideoDeleteRequest) (*schemas.BifrostVideoDeleteResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.VideoDeleteRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) VideoList(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostVideoListRequest) (*schemas.BifrostVideoListResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.VideoListRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) VideoRemix(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostVideoRemixRequest) (*schemas.BifrostVideoGenerationResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.VideoRemixRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) BatchCreate(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostBatchCreateRequest) (*schemas.BifrostBatchCreateResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.BatchCreateRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) BatchList(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostBatchListRequest) (*schemas.BifrostBatchListResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.BatchListRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) BatchRetrieve(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostBatchRetrieveRequest) (*schemas.BifrostBatchRetrieveResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.BatchRetrieveRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) BatchCancel(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostBatchCancelRequest) (*schemas.BifrostBatchCancelResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.BatchCancelRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) BatchDelete(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostBatchDeleteRequest) (*schemas.BifrostBatchDeleteResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.BatchDeleteRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) BatchResults(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostBatchResultsRequest) (*schemas.BifrostBatchResultsResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.BatchResultsRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) FileUpload(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostFileUploadRequest) (*schemas.BifrostFileUploadResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.FileUploadRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) FileList(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostFileListRequest) (*schemas.BifrostFileListResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.FileListRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) FileRetrieve(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostFileRetrieveRequest) (*schemas.BifrostFileRetrieveResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.FileRetrieveRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) FileDelete(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostFileDeleteRequest) (*schemas.BifrostFileDeleteResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.FileDeleteRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) FileContent(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostFileContentRequest) (*schemas.BifrostFileContentResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.FileContentRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) ContainerCreate(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostContainerCreateRequest) (*schemas.BifrostContainerCreateResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ContainerCreateRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) ContainerList(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostContainerListRequest) (*schemas.BifrostContainerListResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ContainerListRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) ContainerRetrieve(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostContainerRetrieveRequest) (*schemas.BifrostContainerRetrieveResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ContainerRetrieveRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) ContainerDelete(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostContainerDeleteRequest) (*schemas.BifrostContainerDeleteResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ContainerDeleteRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) ContainerFileCreate(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostContainerFileCreateRequest) (*schemas.BifrostContainerFileCreateResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ContainerFileCreateRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) ContainerFileList(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostContainerFileListRequest) (*schemas.BifrostContainerFileListResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ContainerFileListRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) ContainerFileRetrieve(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostContainerFileRetrieveRequest) (*schemas.BifrostContainerFileRetrieveResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ContainerFileRetrieveRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) ContainerFileContent(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostContainerFileContentRequest) (*schemas.BifrostContainerFileContentResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ContainerFileContentRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) ContainerFileDelete(_ *schemas.BifrostContext, _ []schemas.Key, _ *schemas.BifrostContainerFileDeleteRequest) (*schemas.BifrostContainerFileDeleteResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.ContainerFileDeleteRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) Passthrough(_ *schemas.BifrostContext, _ schemas.Key, _ *schemas.BifrostPassthroughRequest) (*schemas.BifrostPassthroughResponse, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.PassthroughRequest, provider.GetProviderKey())
}

func (provider *GMIProvider) PassthroughStream(_ *schemas.BifrostContext, _ schemas.PostHookRunner, _ schemas.Key, _ *schemas.BifrostPassthroughRequest) (chan *schemas.BifrostStreamChunk, *schemas.BifrostError) {
	return nil, providerUtils.NewUnsupportedOperationError(schemas.PassthroughStreamRequest, provider.GetProviderKey())
}
