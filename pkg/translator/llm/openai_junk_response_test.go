package llm

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func junkServer(t *testing.T, contentType, body string) *OpenAIClient {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", contentType)
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	c, err := NewOpenAIClient(TranslationConfig{Provider: "github-models", Model: "openai/gpt-4o-mini", APIKey: "k", BaseURL: srv.URL})
	if err != nil {
		t.Fatal(err)
	}
	return c
}

const junkPrompt = "Translate the following text from English to German, preserving formatting:\n\nHello, how are you today?"

// GitHub Models (models.github.ai/inference) answered a real chat request with
// HTTP 200, content-type text/plain and the literal body `OK` (recorded
// 2026-11-24). It must be refused with a typed error naming the cause, never
// surfaced as a JSON-unmarshal accident and never stored as a translation.
func TestOpenAIClient_NonJSON200IsTypedError(t *testing.T) {
	c := junkServer(t, "text/plain", "OK")
	out, err := c.Translate(context.Background(), "Hello", junkPrompt)
	if out != "" || !errors.Is(err, ErrNotChatEndpoint) {
		t.Fatalf("out=%q err=%v, want ErrNotChatEndpoint", out, err)
	}
	if isTextSizeError(err) {
		t.Fatalf("junk-endpoint error must not trigger the size-retry splitter: %v", err)
	}
	if strings.Contains(err.Error(), "k\n") {
		t.Fatal("error leaks request material")
	}
}

// A well-formed chat JSON whose assistant content is only an acknowledgement is
// not a translation of a 60-byte prompt.
func TestOpenAIClient_PlaceholderReplyIsTypedError(t *testing.T) {
	for _, reply := range []string{"OK", "ok.", " Okay! ", "Done", "Understood"} {
		c := junkServer(t, "application/json", `{"choices":[{"message":{"role":"assistant","content":"`+reply+`"}}]}`)
		out, err := c.Translate(context.Background(), "Hello", junkPrompt)
		if out != "" || !errors.Is(err, ErrPlaceholderReply) {
			t.Fatalf("reply %q: out=%q err=%v, want ErrPlaceholderReply", reply, out, err)
		}
	}
}

// Legitimate outputs must be untouched: a real translation, and a genuinely
// tiny source whose correct translation is itself "OK".
func TestOpenAIClient_RealReplyUnaffected(t *testing.T) {
	c := junkServer(t, "application/json", `{"choices":[{"message":{"role":"assistant","content":"Hallo, wie geht es dir heute?"}}]}`)
	out, err := c.Translate(context.Background(), "Hello", junkPrompt)
	if err != nil || out != "Hallo, wie geht es dir heute?" {
		t.Fatalf("out=%q err=%v", out, err)
	}
	c2 := junkServer(t, "application/json", `{"choices":[{"message":{"role":"assistant","content":"OK"}}]}`)
	if out, err := c2.Translate(context.Background(), "OK", "OK"); err != nil || out != "OK" {
		t.Fatalf("short source OK->OK was wrongly rejected: out=%q err=%v", out, err)
	}
}
