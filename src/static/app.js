
// ============================================================
// Banking Support Intelligence - Frontend
// Supports both the original RAG API and the MCP-enabled agent.
// ============================================================

// Existing chat interface elements.
const form = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const sendButton = document.getElementById("send-button");
const messages = document.getElementById("chat-messages");
const sourcesPanel = document.getElementById("sources");
const sourceCount = document.getElementById("source-count");

// New elements for selecting the assistant workflow.
const modeSelector = document.getElementById("chat-mode");
const modeDescription = document.getElementById("mode-description");

// Prevent multiple simultaneous requests.
let busy = false;

// Holds the LangGraph thread ID for the current MCP conversation.
// The same ID must be sent with every subsequent MCP request.
let mcpConversationId = null;


// ============================================================
// Update the description when the assistant mode changes.
// ============================================================

function updateModeDescription() {
    if (modeSelector.value === "mcp") {
        modeDescription.textContent =
            "Agentic workflow with intelligent MCP tool selection.";
    } else {
        modeDescription.textContent =
            "Direct retrieval and grounded answer generation.";
    }
}

modeSelector.addEventListener("change", updateModeDescription);


// ============================================================
// Display messages in the conversation panel.
// ============================================================

function addMessage(text, role = "assistant", meta = "") {
    const message = document.createElement("div");
    message.className = `message ${role}`;

    // textContent prevents model-generated HTML from being executed.
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


// ============================================================
// Display policy source metadata.
// ============================================================

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

        // Display the citation ID and original document path.
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


// ============================================================
// Submit a customer question.
// ============================================================

async function submitQuestion(value) {
    const question = (value ?? questionInput.value).trim();

    if (!question || busy) return;

    if (question.length > 2000) {
        addMessage(
            "Please keep your question within 2,000 characters."
        );
        return;
    }

    // Capture the selected mode before starting the request.
    const selectedMode = modeSelector.value;

    // Each mode uses a different FastAPI endpoint.
    const endpoint = selectedMode === "mcp"
        ? "/mcp/chat"
        : "/chat";

    busy = true;
    sendButton.disabled = true;
    modeSelector.disabled = true;

    addMessage(question, "user");
    questionInput.value = "";

    // Display a loading message while the backend processes the request.
    const loading = addMessage(
        selectedMode === "mcp"
            ? "Agent is evaluating your question and available tools…"
            : "Searching policy evidence and generating an answer…",
        "assistant"
    );

    loading.classList.add("loading-indicator");

    renderSources([]);

    try {
        // Both endpoints receive the question.
        const requestBody = { question };

        // MCP additionally receives the conversation ID.
        // This allows LangGraph to load the correct checkpoint.
        if (selectedMode === "mcp" && mcpConversationId) {
            requestBody.conversation_id = mcpConversationId;
        }

        const response = await fetch(endpoint, {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(requestBody),

            // Allow extra time for hosted LLM calls and MCP processing.
            signal: AbortSignal.timeout(120000)
        });

        const result = await response.json().catch(() => ({}));

        if (!response.ok) {
            const detail = typeof result.detail === "string"
                ? result.detail
                : `Request failed with status ${response.status}`;

            throw new Error(detail);
        }

        loading.remove();

        // Save the conversation ID returned by the MCP API.
        // Subsequent questions will reuse this ID.
        if (
            selectedMode === "mcp" &&
            result.conversation_id
        ) {
            mcpConversationId = result.conversation_id;
        }

        // Build metadata that accurately describes how the answer
        // was generated.
        let answerMeta;

        if (selectedMode === "mcp") {
            answerMeta = result.used_retrieval
                ? "MCP agent retrieved policy evidence through its tools."
                : "MCP agent answered directly without policy retrieval.";
        } else {
            answerMeta = result.abstained
                ? "The answer could not be verified from available policy evidence."
                : "Generated using the standard RAG workflow.";
        }

        addMessage(
            result.answer || "The assistant returned an empty answer.",
            "assistant",
            answerMeta
        );

        // Display sources returned by either backend.
        renderSources(
            Array.isArray(result.sources) ? result.sources : []
        );

        // Show backend notices, if any.
        if (result.error) {
            addMessage(`Notice: ${result.error}`, "assistant");
        }

    } catch (error) {
        loading.remove();

        let errorMessage;

        if (
            error.name === "TimeoutError" ||
            error.name === "AbortError"
        ) {
            errorMessage =
                "The request timed out. Please try again.";
        } else if (error.name === "TypeError") {
            errorMessage =
                "Could not connect to the API. Check that FastAPI is running.";
        } else {
            errorMessage =
                `Unable to get an answer: ${error.message}`;
        }

        addMessage(errorMessage, "assistant");

        // Restore the question so the user can retry.
        questionInput.value = question;

    } finally {
        busy = false;
        sendButton.disabled = false;
        modeSelector.disabled = false;
        questionInput.focus();
    }
}


// ============================================================
// Form submission.
// ============================================================

form.addEventListener("submit", (event) => {
    event.preventDefault();
    submitQuestion();
});


// ============================================================
// Suggested-question cards.
// ============================================================

document.querySelectorAll("[data-question]").forEach((button) => {
    button.addEventListener("click", () => {
        submitQuestion(button.dataset.question);
    });
});


// ============================================================
// Keyboard behaviour.
// Enter submits; Shift + Enter creates a new line.
// ============================================================

questionInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        submitQuestion();
    }
});


// Set the correct description when the page initially loads.
updateModeDescription();
