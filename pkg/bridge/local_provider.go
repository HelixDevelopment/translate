package bridge

import (
	"fmt"
	"net"
	"net/url"
	"strings"

	"digital.vasic.translator/internal/verifier"
)

// LocalProviderOllama is the opt-in local OpenAI-compatible provider id. It is
// reachable ONLY through an explicit `-provider ollama -model <id>`: it is not in
// the verifier's env table, so it is never auto-discovered, never ranked, and the
// default strongest-verified path can never select it (no silent local fallback,
// §11.4.69).
const LocalProviderOllama = "ollama"

const defaultOllamaBaseURL = "http://127.0.0.1:11434/v1"

// IsLocalProvider reports whether id names the opt-in local provider.
func IsLocalProvider(id string) bool { return strings.TrimSpace(id) == LocalProviderOllama }

// ValidateLoopbackBaseURL accepts only http(s) URLs whose host is `localhost`
// or a loopback IP literal (127.0.0.0/8, ::1). A local provider that could be
// pointed at an arbitrary host would send the document (and any key) off-machine
// while being reported as "local"; refuse that rather than trust the operator's
// spelling. Hostnames other than `localhost` are rejected without DNS resolution
// (resolution could change between check and use).
func ValidateLoopbackBaseURL(raw string) error {
	u, err := url.Parse(raw)
	if err != nil || u.Host == "" {
		return fmt.Errorf("bridge: local provider base URL %q is not a valid URL", raw)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return fmt.Errorf("bridge: local provider base URL scheme %q is not http/https", u.Scheme)
	}
	if u.User != nil {
		return fmt.Errorf("bridge: local provider base URL must not embed credentials")
	}
	host := u.Hostname()
	if strings.EqualFold(host, "localhost") {
		return nil
	}
	if ip := net.ParseIP(host); ip != nil && ip.IsLoopback() {
		return nil
	}
	return fmt.Errorf("bridge: local provider base URL host %q is not loopback (localhost, 127.0.0.0/8, ::1); "+
		"refusing to send the document to a non-local host under the local provider", host)
}

// localClient builds the opt-in local client. -model is mandatory (there is no
// verified registry for it and no safe default); the base URL is -base-url, else
// $OLLAMA_BASE_URL, else the default loopback Ollama endpoint, and is validated
// as loopback in every case. Ollama needs no key; the OpenAI-compatible client
// requires a non-empty bearer, so a fixed non-secret placeholder is sent.
func localClient(req ProviderRequest, getenv func(string) string) (llmClient, error) {
	model := strings.TrimSpace(req.Model)
	if model == "" {
		return nil, fmt.Errorf("bridge: provider %q requires an explicit -model (no verified registry for local models)", LocalProviderOllama)
	}
	base := strings.TrimSpace(req.BaseURL)
	if base == "" && getenv != nil {
		base = strings.TrimSpace(getenv("OLLAMA_BASE_URL"))
	}
	if base == "" {
		base = defaultOllamaBaseURL
	}
	if err := ValidateLoopbackBaseURL(base); err != nil {
		return nil, err
	}
	c, err := clientBuild(LocalProviderOllama, model, &verifier.ResolvedProvider{ProviderID: LocalProviderOllama, APIKey: "ollama", BaseURL: base})
	if err != nil {
		return nil, fmt.Errorf("bridge: build %s client: %w", LocalProviderOllama, err)
	}
	return c, nil
}
