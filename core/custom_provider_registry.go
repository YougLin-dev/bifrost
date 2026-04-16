package bifrost

import (
	"sync"

	"github.com/maximhq/bifrost/core/schemas"
)

var customProviderBaseTypes sync.Map

// RegisterCustomProviderBaseType records the base provider for a custom provider key.
func RegisterCustomProviderBaseType(providerKey, baseProvider schemas.ModelProvider) {
	if providerKey == "" || baseProvider == "" || IsStandardProvider(providerKey) {
		return
	}
	customProviderBaseTypes.Store(providerKey, baseProvider)
}

// UnregisterCustomProviderBaseType removes the base-provider mapping for a custom provider key.
func UnregisterCustomProviderBaseType(providerKey schemas.ModelProvider) {
	if providerKey == "" || IsStandardProvider(providerKey) {
		return
	}
	customProviderBaseTypes.Delete(providerKey)
}

// GetBaseProviderType returns the standard base provider behind a provider key.
func GetBaseProviderType(providerKey schemas.ModelProvider) schemas.ModelProvider {
	if providerKey == "" {
		return ""
	}
	if IsStandardProvider(providerKey) {
		return providerKey
	}
	if baseProvider, ok := customProviderBaseTypes.Load(providerKey); ok {
		if typedBaseProvider, ok := baseProvider.(schemas.ModelProvider); ok {
			return typedBaseProvider
		}
	}
	return ""
}

// SyncCustomProviderBaseType keeps the runtime base-provider registry aligned
// with the latest provider configuration.
func SyncCustomProviderBaseType(providerKey schemas.ModelProvider, config *schemas.ProviderConfig) {
	if config != nil && config.CustomProviderConfig != nil && config.CustomProviderConfig.BaseProviderType != "" {
		RegisterCustomProviderBaseType(providerKey, config.CustomProviderConfig.BaseProviderType)
		return
	}
	UnregisterCustomProviderBaseType(providerKey)
}
