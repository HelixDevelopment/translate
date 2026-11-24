package main

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestResolveProviderAPIKeyExplicit(t *testing.T) {
	cfg := &UnifiedConfig{
		Provider: "openai",
		APIKey:   "explicit-key",
	}
	assert.Equal(t, "explicit-key", resolveProviderAPIKey(cfg, "openai"))
}

func TestResolveProviderAPIKeyMismatch(t *testing.T) {
	cfg := &UnifiedConfig{
		Provider: "openai",
		APIKey:   "openai-key",
	}
	// Different provider should not get the explicit key
	assert.Empty(t, resolveProviderAPIKey(cfg, "anthropic"))
}

func TestResolveProviderAPIKeyFromEnv(t *testing.T) {
	t.Setenv("GROQ_API_KEY", "groq-env-key")
	cfg := &UnifiedConfig{Provider: "openai", APIKey: ""}
	assert.Equal(t, "groq-env-key", resolveProviderAPIKey(cfg, "groq"))
}

func TestResolveProviderAPIKeyUnknownProvider(t *testing.T) {
	cfg := &UnifiedConfig{Provider: "unknown", APIKey: ""}
	assert.Empty(t, resolveProviderAPIKey(cfg, "unknown-provider"))
}

func TestResolveProviderAPIKeyPriority(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "env-key")
	cfg := &UnifiedConfig{
		Provider: "openai",
		APIKey:   "flag-key",
	}
	// Explicit flag key takes priority over env var for matching provider
	assert.Equal(t, "flag-key", resolveProviderAPIKey(cfg, "openai"))
}

// -use-verifier resolved keys from a legacy map, so a provider KNOWN to the
// verifier table (nvidia, codestral) but absent from it was reported "API key
// required" while its env var was set (reproduced 2026-11-24).
func TestResolveProviderAPIKey_VerifierTableProviders(t *testing.T) {
	t.Setenv("NVIDIA_API_KEY", "nv-env")
	t.Setenv("CODESTRAL_API_KEY", "cs-env")
	cfg := &UnifiedConfig{Provider: "other"}
	assert.Equal(t, "nv-env", resolveProviderAPIKey(cfg, "nvidia"))
	assert.Equal(t, "cs-env", resolveProviderAPIKey(cfg, "codestral"))
}
