package bridge

import (
	"context"
	"os"
	"testing"

	"digital.vasic.translator/internal/verifier"
	"digital.vasic.translator/internal/verifier/selection"
)

// TestClientForProvider_RoutesToRequestedProvider locks the explicit-`-provider`
// routing contract that the §11.4 unified-translator fix depends on: an EXPLICIT
// provider selection MUST build a client for THAT provider, at THAT provider's
// base URL, with the requested-or-strongest-verified model — NEVER a silent switch
// to a different provider / OpenAI gpt-4 (the reproduced bug: `-provider gemini`
// logged model=gpt-4 and hit OpenAI). It intercepts the single client-construction
// seam (clientBuild) so the (marker, model, baseURL) a build is actually asked for
// is asserted WITHOUT a network call.
func TestClientForProvider_RoutesToRequestedProvider(t *testing.T) {
	// One verified model per provider so the bare-provider (no -model) path can
	// select that provider's strongest verified model.
	b := newTestBridge(keyForAll(),
		verifiedModel("Sao10K/L3-8B-Stheno-v3.2", "novita", "Novita Stheno", 0.92),
		verifiedModel("codestral-2508", "mistral", "Codestral", 0.91),
		verifiedModel("allam-2-7b", "groq", "Groq Allam", 0.90),
		verifiedModel("deepseek-chat", "deepseek", "DeepSeek", 0.85),
		verifiedModel("glm-4.6", "zhipu", "Zhipu GLM", 0.80),
		verifiedModel("gpt-4o", "openai", "OpenAI", 0.70),
	)

	tests := []struct {
		name         string
		req          ProviderRequest
		wantMarker   string
		wantModel    string
		wantBaseHost string // substring the resolved base URL must contain
	}{
		{
			name:         "novita bare provider picks its strongest verified model + base",
			req:          ProviderRequest{ProviderID: "novita"},
			wantMarker:   "novita",
			wantModel:    "Sao10K/L3-8B-Stheno-v3.2",
			wantBaseHost: "api.novita.ai",
		},
		{
			name:         "mistral bare provider",
			req:          ProviderRequest{ProviderID: "mistral"},
			wantMarker:   "mistral",
			wantModel:    "codestral-2508",
			wantBaseHost: "api.mistral.ai",
		},
		{
			name:         "groq bare provider",
			req:          ProviderRequest{ProviderID: "groq"},
			wantMarker:   "groq",
			wantModel:    "allam-2-7b",
			wantBaseHost: "api.groq.com",
		},
		{
			name:         "deepseek bare provider",
			req:          ProviderRequest{ProviderID: "deepseek"},
			wantMarker:   "deepseek",
			wantModel:    "deepseek-chat",
			wantBaseHost: "api.deepseek.com",
		},
		{
			name:         "zhipu bare provider",
			req:          ProviderRequest{ProviderID: "zhipu"},
			wantMarker:   "zhipu",
			wantModel:    "glm-4.6",
			wantBaseHost: "bigmodel.cn",
		},
		{
			name:         "gemini routes to Google's OpenAI-compatible endpoint, NOT OpenAI",
			req:          ProviderRequest{ProviderID: "gemini", Model: "gemini-2.5-flash"},
			wantMarker:   "gemini",
			wantModel:    "gemini-2.5-flash",
			wantBaseHost: "generativelanguage.googleapis.com",
		},
		{
			name:         "explicit -model overrides the verified model",
			req:          ProviderRequest{ProviderID: "groq", Model: "llama-3.3-70b-versatile"},
			wantMarker:   "groq",
			wantModel:    "llama-3.3-70b-versatile",
			wantBaseHost: "api.groq.com",
		},
		{
			name:         "explicit -base-url overrides the resolved base",
			req:          ProviderRequest{ProviderID: "groq", BaseURL: "https://proxy.example.com/v1"},
			wantMarker:   "groq",
			wantModel:    "allam-2-7b",
			wantBaseHost: "proxy.example.com",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var gotMarker, gotModel, gotBase string
			orig := clientBuild
			t.Cleanup(func() { clientBuild = orig })
			clientBuild = func(marker, modelID string, rp *verifier.ResolvedProvider) (llmClient, error) {
				gotMarker = marker
				gotModel = modelID
				gotBase = rp.BaseURL
				return orig(marker, modelID, rp)
			}

			c, err := b.ClientForProvider(context.Background(), tc.req)
			if err != nil {
				t.Fatalf("ClientForProvider(%q) error: %v", tc.req.ProviderID, err)
			}
			if c == nil {
				t.Fatal("ClientForProvider returned nil client")
			}
			if gotMarker != tc.wantMarker {
				t.Errorf("provider marker = %q, want %q (a wrong marker = silent wrong-provider routing)", gotMarker, tc.wantMarker)
			}
			if gotModel != tc.wantModel {
				t.Errorf("model = %q, want %q", gotModel, tc.wantModel)
			}
			if !containsSub(gotBase, tc.wantBaseHost) {
				t.Errorf("base URL = %q, want it to contain %q (wrong host = request goes to the wrong provider)", gotBase, tc.wantBaseHost)
			}
			// The reported provider name must be the REQUESTED provider, never the
			// bare *OpenAIClient literal "openai" (the misleading-log half of the bug).
			if got := c.GetProviderName(); got != tc.wantMarker {
				t.Errorf("GetProviderName() = %q, want %q", got, tc.wantMarker)
			}
		})
	}
}

// TestClientForProvider_NoSilentFallback proves the no-silent-fallback rule
// (§11.4.69): a bare explicit provider with NO verified model and NO -model is an
// honest hard error — NOT a silent switch to another provider's strongest model.
func TestClientForProvider_NoSilentFallback(t *testing.T) {
	b := newTestBridge(keyForAll(),
		verifiedModel("allam-2-7b", "groq", "Groq Allam", 0.90), // only groq is verified
	)
	_, err := b.ClientForProvider(context.Background(), ProviderRequest{ProviderID: "gemini"})
	if err == nil {
		t.Fatal("expected an honest error for provider with no verified model and no -model, got nil (silent fallback)")
	}
	if !containsSub(err.Error(), "gemini") || !containsSub(err.Error(), "no verified model") {
		t.Errorf("error %q should name the misconfigured provider and the missing-model cause", err.Error())
	}
}

// TestClientForProvider_MissingKeyHonestError proves a KNOWN provider with no key
// (no -api-key and the *_API_KEY env var unset) yields an honest missing-key error
// naming the provider — never an empty-key client and never a silent fallback.
func TestClientForProvider_MissingKeyHonestError(t *testing.T) {
	noKeys := func(string) string { return "" } // no env keys at all
	b := newTestBridge(noKeys,
		verifiedModel("allam-2-7b", "groq", "Groq Allam", 0.90),
	)
	_, err := b.ClientForProvider(context.Background(), ProviderRequest{ProviderID: "groq"})
	if err == nil {
		t.Fatal("expected an honest missing-key error, got nil")
	}
	if !containsSub(err.Error(), "groq") || !containsSub(err.Error(), "API key") {
		t.Errorf("error %q should name the provider and the missing API key", err.Error())
	}
}

// TestTranslatorForProvider_WrapsClient proves the translator surface is built for
// the requested provider (delegates to ClientForProvider) so the unified-translator
// explicit-provider path gets a translator.Translator naming the right provider.
func TestTranslatorForProvider_WrapsClient(t *testing.T) {
	b := newTestBridge(keyForAll(),
		verifiedModel("codestral-2508", "mistral", "Codestral", 0.91),
	)
	tr, err := b.TranslatorForProvider(context.Background(), ProviderRequest{
		ProviderID: "mistral",
		Task:       selection.TaskRequirements{SourceLang: "en", TargetLang: "ru"},
	})
	if err != nil {
		t.Fatalf("TranslatorForProvider error: %v", err)
	}
	if got := tr.GetName(); got != "mistral" {
		t.Errorf("translator GetName() = %q, want mistral", got)
	}
}

func containsSub(s, sub string) bool {
	return len(sub) == 0 || (len(s) >= len(sub) && indexOf(s, sub) >= 0)
}

func indexOf(s, sub string) int {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return i
		}
	}
	return -1
}

// TestClientForProvider_CapacityGapProviders locks the engine-side gaps the i18n
// provider survey found (2026-11-24): cohere and codestral could not be routed at
// all, and nvidia (no verified model because discovery capped on dead models)
// failed with "no verified model". A bare provider with NO verified model must
// now pick its recorded translation model and the right base URL.
func TestClientForProvider_CapacityGapProviders(t *testing.T) {
	b := newTestBridge(keyForAll()) // nothing verified at all
	for _, tc := range []struct{ id, model, host string }{
		{"nvidia", "nvidia/riva-translate-4b-instruct-v2", "integrate.api.nvidia.com"},
		{"cohere", "command-a-translate-08-2025", "api.cohere.com/compatibility/v1"},
		{"codestral", "codestral-latest", "codestral.mistral.ai"},
	} {
		var gotModel, gotBase string
		orig := clientBuild
		clientBuild = func(m, id string, rp *verifier.ResolvedProvider) (llmClient, error) {
			gotModel, gotBase = id, rp.BaseURL
			return orig(m, id, rp)
		}
		_, err := b.ClientForProvider(context.Background(), ProviderRequest{ProviderID: tc.id})
		clientBuild = orig
		if err != nil {
			t.Fatalf("%s: %v", tc.id, err)
		}
		if gotModel != tc.model || !containsSub(gotBase, tc.host) {
			t.Errorf("%s: model=%q base=%q, want %q @ %q", tc.id, gotModel, gotBase, tc.model, tc.host)
		}
	}
	// A provider with no recorded default still hard-errors (no guessing).
	if _, err := b.ClientForProvider(context.Background(), ProviderRequest{ProviderID: "groq"}); err == nil {
		t.Fatal("groq has no preferred model and nothing verified: want honest error")
	}
}

// Open with SkipVerification must succeed with NO provider keys and NO store
// (reproduced RED 2026-11-24: codestral + -model + valid key failed with "no
// models met the verification threshold" because Open insisted on verifying).
func TestOpen_SkipVerificationNeedsNoKeysNoStore(t *testing.T) {
	dir := t.TempDir()
	b, err := Open(context.Background(), Options{
		SkipVerification: true,
		DBPath:           dir + "/must-not-be-created/v.db",
		Getenv:           func(k string) string { return map[string]string{"CODESTRAL_API_KEY": "k"}[k] },
	})
	if err != nil {
		t.Fatalf("Open(SkipVerification) = %v", err)
	}
	if b.Source() != "unverified-explicit" {
		t.Fatalf("source = %q", b.Source())
	}
	if _, serr := os.Stat(dir + "/must-not-be-created"); serr == nil {
		t.Fatal("SkipVerification touched the verified-models store directory")
	}
	if _, err := b.ClientForProvider(context.Background(), ProviderRequest{ProviderID: "codestral", Model: "codestral-latest"}); err != nil {
		t.Fatalf("explicit provider+model on a skip-verification bridge: %v", err)
	}
	// The strict default (no SkipVerification, no keys) still hard-errors.
	if _, err := Open(context.Background(), Options{DBPath: dir + "/x/v.db", Getenv: func(string) string { return "" }}); err == nil {
		t.Fatal("strict Open with no provider keys must still error")
	}
}
