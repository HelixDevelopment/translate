package bridge

import (
	"context"
	"strings"
	"testing"

	"digital.vasic.translator/internal/verifier"
)

func TestValidateLoopbackBaseURL(t *testing.T) {
	ok := []string{"http://127.0.0.1:11434/v1", "http://localhost:11434/v1", "https://127.1.2.3/v1", "http://[::1]:11434/v1", "http://LOCALHOST/v1"}
	bad := []string{"http://192.168.1.10:11434/v1", "http://example.com/v1", "http://127.0.0.1.evil.com/v1",
		"http://user:pw@127.0.0.1/v1", "ftp://127.0.0.1/v1", "127.0.0.1:11434", "", "http://0.0.0.0:11434/v1", "http://[::]:11434"}
	for _, u := range ok {
		if err := ValidateLoopbackBaseURL(u); err != nil {
			t.Errorf("%q rejected: %v", u, err)
		}
	}
	for _, u := range bad {
		if err := ValidateLoopbackBaseURL(u); err == nil {
			t.Errorf("%q accepted, must be refused", u)
		}
	}
}

func TestClientForProvider_OllamaOptInLoopbackOnly(t *testing.T) {
	b := newTestBridge(func(string) string { return "" }) // no keys, nothing verified
	var gotBase, gotModel, gotKey string
	orig := clientBuild
	t.Cleanup(func() { clientBuild = orig })
	clientBuild = func(m, id string, rp *verifier.ResolvedProvider) (llmClient, error) {
		gotBase, gotModel, gotKey = rp.BaseURL, id, rp.APIKey
		return orig(m, id, rp)
	}
	c, err := b.ClientForProvider(context.Background(), ProviderRequest{ProviderID: "ollama", Model: "qwen2.5:3b"})
	if err != nil {
		t.Fatal(err)
	}
	if gotBase != defaultOllamaBaseURL || gotModel != "qwen2.5:3b" || gotKey == "" || c.GetProviderName() != "ollama" {
		t.Fatalf("base=%q model=%q key-set=%v name=%q", gotBase, gotModel, gotKey != "", c.GetProviderName())
	}
	// -model is mandatory.
	if _, err := b.ClientForProvider(context.Background(), ProviderRequest{ProviderID: "ollama"}); err == nil || !strings.Contains(err.Error(), "-model") {
		t.Fatalf("missing -model must be an honest error, got %v", err)
	}
	// A non-loopback -base-url is refused BEFORE any client is built.
	built := false
	clientBuild = func(m, id string, rp *verifier.ResolvedProvider) (llmClient, error) {
		built = true
		return orig(m, id, rp)
	}
	if _, err := b.ClientForProvider(context.Background(), ProviderRequest{ProviderID: "ollama", Model: "x", BaseURL: "http://10.0.0.5:11434/v1"}); err == nil || built {
		t.Fatalf("non-loopback accepted (err=%v built=%v)", err, built)
	}
}

// The default path must never pick the local provider: it is not an env-table
// provider and is not OpenAI-compatible-listed.
func TestOllamaIsNotInDefaultProviderTables(t *testing.T) {
	if openAICompatibleProviders["ollama"] || verifier.EnvVarForProvider("ollama") != "" {
		t.Fatal("ollama leaked into the auto-discovered provider tables")
	}
}
