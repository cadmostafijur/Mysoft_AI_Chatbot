/**
 * Mysoft Heaven AI Chatbot — Frontend Logic
 *
 * Handles:
 *  - Sending questions to the /chat API endpoint
 *  - Rendering chat messages (user + bot)
 *  - Session management for conversation memory
 *  - Typing indicators and UI state
 *  - Suggestion chips for quick questions
 *  - Mobile sidebar toggle
 */

// ── Configuration ─────────────────────────────────────────────────
const API_BASE = window.location.origin; // Same origin when served by FastAPI
const CHAT_ENDPOINT = `${API_BASE}/chat`;
const HEALTH_ENDPOINT = `${API_BASE}/health`;

// ── State ─────────────────────────────────────────────────────────
let sessionId = generateSessionId();
let isWaiting = false;

// ── DOM Elements ──────────────────────────────────────────────────
const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const welcomeScreen = document.getElementById("welcomeScreen");
const btnNewChat = document.getElementById("btnNewChat");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const confidenceBadge = document.getElementById("confidenceBadge");
const mobileMenuBtn = document.getElementById("mobileMenuBtn");

// ── Initialization ────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  setupEventListeners();
  chatInput.focus();
});

function setupEventListeners() {
  // Send on button click
  sendBtn.addEventListener("click", handleSend);

  // Send on Enter (Shift+Enter for new line)
  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });

  // Enable/disable send button based on input
  chatInput.addEventListener("input", () => {
    sendBtn.disabled = chatInput.value.trim() === "" || isWaiting;
    autoResizeTextarea();
  });

  // Suggestion chips
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const question = chip.getAttribute("data-question");
      if (question) {
        chatInput.value = question;
        handleSend();
      }
    });
  });

  // New chat
  btnNewChat.addEventListener("click", startNewChat);

  // Mobile sidebar
  mobileMenuBtn.addEventListener("click", toggleSidebar);
}

// ── Health Check ──────────────────────────────────────────────────
async function checkHealth() {
  try {
    const res = await fetch(HEALTH_ENDPOINT);
    if (res.ok) {
      statusDot.className = "status-dot online";
      statusText.textContent = "Online";
    } else {
      throw new Error("Not healthy");
    }
  } catch {
    statusDot.className = "status-dot offline";
    statusText.textContent = "Offline";
  }
}

// ── Send Message ──────────────────────────────────────────────────
async function handleSend() {
  const question = chatInput.value.trim();
  if (!question || isWaiting) return;

  // Hide welcome screen
  if (welcomeScreen) {
    welcomeScreen.style.display = "none";
  }

  // Add user message
  appendMessage("user", question);
  chatInput.value = "";
  chatInput.style.height = "auto";
  sendBtn.disabled = true;
  isWaiting = true;

  // Show typing indicator
  const typingEl = showTypingIndicator();

  try {
    const response = await fetch(CHAT_ENDPOINT, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question,
        company_id: "mysoft_heaven",
        session_id: sessionId,
      }),
    });

    // Remove typing indicator
    removeTypingIndicator(typingEl);

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${response.status})`);
    }

    const data = await response.json();

    // Update session ID from response
    if (data.session_id) {
      sessionId = data.session_id;
    }

    // Show confidence badge
    showConfidence(data.confidence);

    // Append bot response
    appendMessage("bot", data.answer, {
      confidence: data.confidence,
      sources: data.sources,
      fallback: data.fallback,
    });
  } catch (error) {
    removeTypingIndicator(typingEl);
    appendMessage("bot", `Sorry, I encountered an error: ${error.message}. Please make sure the backend server is running.`, {
      fallback: true,
    });
  }

  isWaiting = false;
  sendBtn.disabled = chatInput.value.trim() === "";
  chatInput.focus();
}

// ── Render Messages ───────────────────────────────────────────────
function appendMessage(role, text, meta = {}) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = role === "user" ? "U" : "M";

  const content = document.createElement("div");
  content.className = "message-content";

  const bubble = document.createElement("div");
  bubble.className = `message-bubble${meta.fallback ? " fallback" : ""}`;
  bubble.innerHTML = formatMessage(text);

  content.appendChild(bubble);

  // Add source info for bot messages
  if (role === "bot" && meta.sources && meta.sources.length > 0) {
    const metaDiv = document.createElement("div");
    metaDiv.className = "message-meta";

    const uniqueFiles = [...new Set(meta.sources.map((s) => s.filename))];
    uniqueFiles.forEach((file) => {
      const sourceTag = document.createElement("span");
      sourceTag.className = "message-source";
      sourceTag.textContent = file;
      metaDiv.appendChild(sourceTag);
    });

    content.appendChild(metaDiv);
  }

  messageDiv.appendChild(avatar);
  messageDiv.appendChild(content);
  chatMessages.appendChild(messageDiv);

  // Scroll to bottom
  scrollToBottom();
}

function formatMessage(text) {
  // Basic markdown-like formatting
  let html = escapeHtml(text);

  // Bold: **text**
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

  // Bullet lists: lines starting with - or *
  html = html.replace(/^[\-\*]\s+(.+)$/gm, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>\n?)+/g, "<ul>$&</ul>");

  // Numbered lists: lines starting with 1. 2. etc.
  html = html.replace(/^\d+\.\s+(.+)$/gm, "<li>$1</li>");

  // Line breaks
  html = html.replace(/\n/g, "<br>");

  // Wrap in paragraph if no block elements
  if (!html.includes("<ul>") && !html.includes("<ol>")) {
    html = `<p>${html}</p>`;
  }

  return html;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ── Typing Indicator ──────────────────────────────────────────────
function showTypingIndicator() {
  const messageDiv = document.createElement("div");
  messageDiv.className = "message bot";
  messageDiv.id = "typingMessage";

  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = "M";

  const content = document.createElement("div");
  content.className = "message-content";

  const bubble = document.createElement("div");
  bubble.className = "message-bubble";
  bubble.innerHTML = `
    <div class="typing-indicator">
      <div class="dot"></div>
      <div class="dot"></div>
      <div class="dot"></div>
    </div>
  `;

  content.appendChild(bubble);
  messageDiv.appendChild(avatar);
  messageDiv.appendChild(content);
  chatMessages.appendChild(messageDiv);
  scrollToBottom();

  return messageDiv;
}

function removeTypingIndicator(el) {
  if (el && el.parentNode) {
    el.parentNode.removeChild(el);
  }
}

// ── Confidence Badge ──────────────────────────────────────────────
function showConfidence(level) {
  confidenceBadge.style.display = "inline-block";
  confidenceBadge.className = `confidence-badge ${level}`;

  const labels = {
    high: "High Confidence",
    low: "Low Confidence",
    none: "No Match",
  };
  confidenceBadge.textContent = labels[level] || level;

  // Auto-hide after 8 seconds
  setTimeout(() => {
    confidenceBadge.style.display = "none";
  }, 8000);
}

// ── New Chat ──────────────────────────────────────────────────────
function startNewChat() {
  sessionId = generateSessionId();
  chatMessages.innerHTML = "";
  if (welcomeScreen) {
    chatMessages.appendChild(welcomeScreen);
    welcomeScreen.style.display = "flex";
  }
  confidenceBadge.style.display = "none";
  chatInput.value = "";
  chatInput.focus();
  closeSidebar();
}

// ── Helpers ───────────────────────────────────────────────────────
function generateSessionId() {
  return "sess_" + Date.now().toString(36) + "_" + Math.random().toString(36).substr(2, 6);
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  });
}

function autoResizeTextarea() {
  chatInput.style.height = "auto";
  chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
}

// ── Mobile Sidebar ────────────────────────────────────────────────
function toggleSidebar() {
  const sidebar = document.querySelector(".sidebar");
  sidebar.classList.toggle("open");

  let overlay = document.querySelector(".sidebar-overlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.className = "sidebar-overlay";
    overlay.addEventListener("click", closeSidebar);
    document.body.appendChild(overlay);
  }
  overlay.classList.toggle("active");
}

function closeSidebar() {
  const sidebar = document.querySelector(".sidebar");
  sidebar.classList.remove("open");
  const overlay = document.querySelector(".sidebar-overlay");
  if (overlay) overlay.classList.remove("active");
}

// ── Periodic Health Check ─────────────────────────────────────────
setInterval(checkHealth, 30000);
