
const form = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const sendButton = document.getElementById("send-button");
const messages = document.getElementById("chat-messages");
const sourcesPanel = document.getElementById("sources");
const sourceCount = document.getElementById("source-count");

let busy = false;

function addMessage(text, role = "assistant", meta = "") {
    const message = document.createElement("div");
    message.className = `message ${role}`;

    // Use textContent so model output cannot inject HTML.
    message.textContent = text;

    if (meta) {
        const metadata = document.createElement("small");
        metadata.className = "message-meta";
        metadata.textContent = meta;
        message.appendChild(metadata);
    }

    messages.appendChild(message);
    messages.scrollTop = messages.scrollHeight;

    return message;
}

function renderSources(sources = []) {
    sourcesPanel.replaceChildren();

    if (sourceCount) {
        sourceCount.textContent = `${sources.length} source${
            sources.length === 1 ? "" : "s"
        }`;
    }

    if (!sources.length) {
        const empty = document.createElement("p");
        empty.className = "sources-empty";
        empty.textContent =
            "No supporting policy sources were returned.";
        sourcesPanel.appendChild(empty);
        return;
    }

    for (const source of sources) {
        const item = document.createElement("details");
        item.className = "source-item";

        const title = document.createElement("summary");
        title.textContent =
            `[${source.citation_id || "Source"}] ` +
            (source.source || "Policy document");

        const details = document.createElement("div");
        details.className = "source-details";

        const documentId = document.createElement("p");
        documentId.textContent =
            `Document ID: ${source.document_id || "Unknown"}`;

        const distance = document.createElement("p");
        distance.textContent =
            `Retrieval distance: ${
                typeof source.distance === "number"
                    ? source.distance.toFixed(3)
                    : "Not available"
            }`;

        const note = document.createElement("p");
        note.textContent =
            "The API currently returns source metadata, not policy text excerpts.";

        details.append(documentId, distance, note);
        item.append(title, details);
        sourcesPanel.appendChild(item);
    }
}

async function submitQuestion(value) {
    const question = (value ?? questionInput.value).trim();

    if (!question || busy) return;

    if (question.length > 2000) {
        addMessage(
            "Please keep your question within 2,000 characters."
        );
        return;
    }

    busy = true;
    sendButton.disabled = true;

    addMessage(question, "user");
    questionInput.value = "";

    const loading = addMessage(
        "Searching policy evidence and generating an answer…",
        "assistant"
    );

    renderSources([]);

    try {
        const response = await fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ question }),
            signal: AbortSignal.timeout(90000)
        });

        const result = await response.json().catch(() => ({}));

        if (!response.ok) {
            const detail = typeof result.detail === "string"
                ? result.detail
                : `Request failed with status ${response.status}`;

            throw new Error(detail);
        }

        loading.remove();

        addMessage(
            result.answer || "The assistant returned an empty answer.",
            "assistant",
            result.abstained
                ? "The answer could not be verified from available policy evidence."
                : "Generated using retrieved policy evidence."
        );

        renderSources(
            Array.isArray(result.sources) ? result.sources : []
        );

        if (result.error) {
            addMessage(`Notice: ${result.error}`, "assistant");
        }
    } catch (error) {
        loading.remove();

        let message;

        if (error.name === "TimeoutError") {
            message =
                "The request timed out. Please try again.";
        } else if (error.name === "TypeError") {
            message =
                "Could not connect to the API. Check that the FastAPI server is running.";
        } else {
            message = `Unable to get an answer: ${error.message}`;
        }

        addMessage(message, "assistant");
        questionInput.value = question;
    } finally {
        busy = false;
        sendButton.disabled = false;
        questionInput.focus();
    }
}

form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitQuestion();
});

// Connect the suggested-question cards in the HTML.
document.querySelectorAll("[data-question]").forEach((button) => {
    button.addEventListener("click", () => {
        submitQuestion(button.dataset.question);
    });
});

// Submit with Enter; use Shift+Enter for a new line.
questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        submitQuestion();
    }
});
