
const form = document.getElementById("chat-form");
const questionInput = document.getElementById("question");
const sendButton = document.getElementById("send-button");
const messages = document.getElementById("chat-messages");
const sourcesPanel = document.getElementById("sources");

let busy = false;

function addMessage(text, role = "assistant") {
    const message = document.createElement("div");
    message.className = `message ${role}`;
    message.textContent = text;
    messages.appendChild(message);
    messages.scrollTop = messages.scrollHeight;
    return message;
}

function renderSources(sources) {
    sourcesPanel.replaceChildren();

    if (!sources.length) {
        const empty = document.createElement("p");
        empty.textContent = "No supporting sources were returned.";
        sourcesPanel.appendChild(empty);
        return;
    }

    for (const source of sources) {
        const item = document.createElement("div");
        item.className = "source-item";

        const title = document.createElement("strong");
        title.textContent =
            `[${source.citation_id || "Source"}] ${source.source || "Policy document"}`;

        const details = document.createElement("small");
        details.textContent =
            `Document: ${source.document_id || "Unknown"} · ` +
            `Distance: ${
                typeof source.distance === "number"
                    ? source.distance.toFixed(3)
                    : "N/A"
            }`;

        item.append(title, details);
        sourcesPanel.appendChild(item);
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const question = questionInput.value.trim();

    if (!question || busy) return;

    busy = true;
    sendButton.disabled = true;

    addMessage(question, "user");
    questionInput.value = "";

    const loading = addMessage("Searching policy evidence…");
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
            throw new Error(
                result.detail || `Request failed with status ${response.status}`
            );
        }

        loading.remove();

        addMessage(
            result.answer || "The assistant returned an empty answer."
        );

        renderSources(
            Array.isArray(result.sources) ? result.sources : []
        );

        if (result.error) {
            addMessage(`Notice: ${result.error}`);
        }
    } catch (error) {
        loading.remove();

        if (error.name === "TimeoutError") {
            addMessage(
                "The request took too long. Please try again."
            );
        }
        else {
            addMessage(
                `Unable to get an answer: ${error.message}`
            );
        }

        questionInput.value = question;
        
    } finally {
        busy = false;
        sendButton.disabled = false;
        questionInput.focus();
    }
});
