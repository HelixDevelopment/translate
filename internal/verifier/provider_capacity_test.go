package verifier

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

// nvidiaCatalog is an excerpt of the REAL alphabetical /v1/models id list returned
// by integrate.api.nvidia.com on 2026-11-24 (ids recorded verbatim). Its first
// three entries are not callable chat models (real chat calls: 404 / 422 / 404),
// which is why the provider never verified before candidate ordering existed.
var nvidiaCatalog = []string{
	"bigcode/starcoder2-15b", "fastino/gliner2-pii", "google/gemma-3-12b-it",
	"google/gemma-3-4b-it", "google/gemma-4-31b-it", "meta/muse-glimmer-30b",
	"mistralai/mistral-large-4-1t-a52b", "nvidia/riva-translate-4b-instruct-v2",
	"z-ai/glm-5.3-flash",
}

func TestOrderCandidates_PreferredFirstNeverInjected(t *testing.T) {
	got := orderCandidates("nvidia", nvidiaCatalog)
	if got[0] != "nvidia/riva-translate-4b-instruct-v2" {
		t.Fatalf("first candidate = %q, want the translation model", got[0])
	}
	if len(got) != len(nvidiaCatalog) {
		t.Fatalf("len = %d want %d (reorder only)", len(got), len(nvidiaCatalog))
	}
	// A preferred id absent from the live catalog must NOT be fabricated.
	if only := orderCandidates("cohere", []string{"embed-v4.0"}); !reflect.DeepEqual(only, []string{"embed-v4.0"}) {
		t.Fatalf("fabricated candidate: %v", only)
	}
	if g := orderCandidates("groq", []string{"b", "a"}); !reflect.DeepEqual(g, []string{"b", "a"}) {
		t.Fatalf("unknown provider reordered: %v", g)
	}
}

// The cap of 3 is what hid the working NVIDIA model: prove it survives the cap.
func TestRunVerification_PreferredModelSurvivesCap(t *testing.T) {
	mux := http.NewServeMux()
	mux.HandleFunc("/models", func(w http.ResponseWriter, r *http.Request) {
		type m struct {
			ID string `json:"id"`
		}
		var d []m
		for _, id := range nvidiaCatalog {
			d = append(d, m{id})
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"data": d})
	})
	mux.HandleFunc("/chat/completions", func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model string `json:"model"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		if req.Model != "nvidia/riva-translate-4b-instruct-v2" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"Hallo"}}]}`))
	})
	srv := httptest.NewServer(mux)
	defer srv.Close()

	sink := newMemSink()
	res, err := RunVerification(context.Background(), NewRegistry(), NewPipeline(), sink,
		[]ProviderConfig{{ID: "nvidia", APIKey: "k", BaseURL: srv.URL}},
		RunOptions{MaxModelsPerProvider: 3, MinScoreToPersist: 0})
	if err != nil {
		t.Fatal(err)
	}
	if res.Providers[0].CandidateModels[0] != "nvidia/riva-translate-4b-instruct-v2" {
		t.Fatalf("candidates = %v", res.Providers[0].CandidateModels)
	}
	if sink.count() != 1 {
		t.Fatalf("verified+persisted = %d, want 1 (the translation model)", sink.count())
	}
}

func TestEnvSpecs_CohereCodestralAppendedAndResolvable(t *testing.T) {
	want := map[string]string{
		"cohere":    "https://api.cohere.com/compatibility/v1",
		"codestral": "https://codestral.mistral.ai/v1",
	}
	for id, base := range want {
		if s, ok := providerByID[id]; !ok || s.BaseURL != base {
			t.Errorf("%s spec = %+v ok=%v, want base %s", id, s, ok, base)
		}
	}
	if EnvVarForProvider("cohere") != "COHERE_API_KEY" || EnvVarForProvider("codestral") != "CODESTRAL_API_KEY" ||
		EnvVarForProvider("nvidia") != "NVIDIA_API_KEY" || EnvVarForProvider("nope") != "" {
		t.Fatal("EnvVarForProvider mapping wrong")
	}
	// numericProviderIndex is positional/append-only: the original 16 must not move.
	if envProviderSpecs[0].ID != "openai" || envProviderSpecs[15].ID != "gemini" {
		t.Fatalf("existing positional ids moved: %s .. %s", envProviderSpecs[0].ID, envProviderSpecs[15].ID)
	}
	if got := PreferredModelFor("nvidia"); got != "nvidia/riva-translate-4b-instruct-v2" {
		t.Fatalf("PreferredModelFor(nvidia) = %q", got)
	}
}
