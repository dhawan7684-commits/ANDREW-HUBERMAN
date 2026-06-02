# Project Blueprint: Dr. Andrew Huberman Digital Twin

**Document Type:** Software Design & Architectural Case Study  
**Classification:** Engineering Reference — Internal Development Record  
**Stack:** Python · LangChain · Streamlit · Google Gemini API

---

## 1. Executive Summary

The Dr. Andrew Huberman Digital Twin is a production-grade conversational AI application built to simulate the intellectual persona, scientific communication style, and knowledge domain of Dr. Andrew Huberman — neuroscientist, professor, and host of the Huberman Lab podcast. The system was engineered using LangChain as the orchestration layer, Google Gemini (`gemini-2.5-flash`) as the underlying language model, and Streamlit as the reactive user interface framework.

**Core Objective:** Deliver real-time, science-grounded human optimization protocols with extreme persona fidelity, deep contextual memory, and zero immersion-breaking AI disclaimers. The twin must respond as Dr. Huberman would — with authority, precision, and biological depth — while dynamically adapting its guidance to each user's unique profile and the exact moment in time they are engaging with it.

The platform achieves this through three mutually reinforcing engineering pillars:

- **Persistent Memory Extraction** — A parallel background process that silently builds a structured profile of the user across the conversation.
- **Live Circadian Clock Integration** — The system captures and injects the user's real-time local timestamp into every LLM invocation, enabling biologically accurate, time-specific protocol recommendations.
- **Strict Persona Enforcement** — A rigorously structured system prompt matrix that eliminates generic AI deflections and forces all responses through the lens of Dr. Huberman's scientific communication framework.

---

## 2. User Interface Design Decisions

### 2.1 The Dual-Panel UI Matrix

The interface is organized as an asymmetric master column layout with a ratio of 1 unit to 3 units of horizontal screen space. This structural decision was deliberate and serves a critical functional purpose: it cleanly separates the *meta-layer* of the interaction (what the system knows about you) from the *primary layer* (the active conversation itself).

Without this separation, the conversational feed becomes a dual-purpose display — housing both the dialogue and incremental memory confirmations — which creates cognitive noise and wastes vertical scroll space. By housing memory in a fixed left panel and conversation in the right, each area can optimize for its own purpose independently.

The gap between the two columns is set to `large`, which introduces sufficient visual breathing room to make the separation feel structural and intentional rather than accidental.

### 2.2 Persistent Memory Panel — Left Column

The left column hosts the **User Insights & Memory** panel. This is a live, file-backed display that reads from a local text file (`user_profile.txt`) and renders its contents directly into the Streamlit interface on every rerender cycle.

The rationale for placing this on the left margin — the natural starting point of the user's reading flow — is that it immediately establishes the system's awareness. A user who mentions their name or a dietary preference will see it captured and surfaced within the next render, reinforcing the perception of an intelligent, attentive agent.

Critically, this also eliminates the need for the twin itself to verbally acknowledge profile captures within the conversation stream ("Got it, I've noted that you..."), which would fragment the scientific dialogue and reduce immersion. The memory panel provides that feedback loop silently, passively, and visually.

The panel also houses the session reset control — a **Reset Chat & Wipe Memory** button — which triggers a complete teardown of both the conversation history and the on-disk profile, ensuring users always have explicit control over their data.

### 2.3 Single-Image Header Realignment

The initial header layout was structured as a three-column container designed to accommodate two image badges flanking the application title. In practice, only a single portrait image was actively maintained (`left_image.png`), leaving the rightmost column empty and creating a visually unbalanced, asymmetric title block that felt cluttered.

The resolution was a full structural replacement: the header was redesigned as a clean two-column block using a `[1, 5]` column ratio. The portrait image is anchored to the narrow left column, and the title typography stack — application name, subtitle caption, and horizontal rule — occupies the full remaining five-unit width.

This change does more than solve an aesthetic problem. It removes an architectural assumption baked into the layout code (that two images will always be present) and replaces it with a stable, single-image contract. The result is a professional, magazine-style header that scales correctly regardless of the image dimensions supplied.

---

## 3. Engineering Approach & Core Logic

### 3.1 Asymmetric Background Memory Extraction

Every user message submitted to the twin triggers two simultaneous processes: the primary response generation pipeline, and an independent, lightweight background extraction routine. These two pipelines are decoupled by design — the extraction task does not delay or block the response pipeline in any way.

The extraction routine sends the raw user message to the same LLM with a tightly scoped, instruction-only prompt designed to identify and isolate personal identity data. The extraction targets four data categories:

- **Core identity indicators** — Name, age, occupation, professional title
- **Lifestyle and behavioral patterns** — Sleep schedules, light exposure routines, technology habits
- **Dietary context** — Food preferences, dietary restrictions, eating windows, notable snacks or meals
- **Physical constraints and tools** — Injuries, equipment, training splits, physical limitations

The output of the extraction is a single, third-person bullet point (e.g., `* Name: Laksh Dhawan`) or the literal string `NONE` if no personal information is detected. If a valid fact is returned, the system reads the existing profile file, checks for duplicate entries, and appends the new fact only if it is genuinely new. This deduplication logic prevents the memory log from accumulating redundant entries across a long session.

All exceptions within this routine are silently caught and suppressed. The extraction task is purely additive — it enriches the session but is never allowed to interrupt the user's experience if it fails.

### 3.2 Real-World Timeline Awareness

At the precise moment a user submits a message, the application captures the system clock using Python's `datetime` module and formats it into a human-readable timestamp string (e.g., `08:30 AM (Monday)`). This timestamp is surfaced in two locations: as a visual caption beneath the user's message bubble in the chat feed, and embedded directly into the input string passed to the LLM.

The timestamp is wrapped in a structured context block that explicitly instructs the model to evaluate the message through a temporal lens. If the user's message involves a behavior — drinking coffee, eating, beginning a workout, winding down for sleep — the model can cross-reference that behavior against the current time and respond with biologically accurate, time-specific guidance.

This architecture enables a qualitatively different category of response. A message stating "I'm about to have my morning coffee" sent at 6:30 AM triggers a caffeine delay recommendation grounded in cortisol spike biology. The same message sent at 2:00 PM generates a different response calibrated to adenosine clearance and afternoon fatigue windows. The time context is not decoration; it is a first-class input to the reasoning pipeline.

### 3.3 Automated Session Lifecycle Management

The application enforces a strict one-session-per-profile contract. Session state is managed through Streamlit's native `st.session_state` dictionary, which is inherently ephemeral — it exists only for the duration of a single browser tab's lifecycle.

On every page load, the application checks whether the `messages` key exists in session state. If it does not — indicating either a fresh tab launch or a full browser reload — the application immediately executes a cleanup hook that deletes the `user_profile.txt` file from disk if it exists. This ensures that profile data from a prior session cannot leak into a new one, which would corrupt both the memory display and the LLM's contextual understanding of who it is speaking with.

This design trades long-term cross-session persistence for strict session hygiene. The decision reflects a deliberate prioritization: an immersive, accurate, single-session experience is more valuable than a persistent but potentially contaminated multi-session memory. Users who require persistence can treat the memory panel as a reference and carry relevant context forward manually.

---

## 4. Chronological Log of Roadblocks & System Bug Fixes

### 4.1 The Context Amnesia Loop

**Symptom:** In early builds, the agent would break character within a single conversation when asked to recall a user detail shared just two or three turns prior. If a user stated their name in turn one and referenced it in turn three, the agent would respond as if it had no record of the information, sometimes explicitly stating it could not access personal data.

**Root Cause:** The initial system prompt gave equal or higher priority to the absence of a fact in the vector database as evidence that the fact was unknown. The prompt hierarchy did not explicitly instruct the model to treat the active `chat_history` array as a higher-authority source of truth than any external retrieval layer.

**Resolution:** The system prompt matrix was restructured to formally declare the active conversation history as the supreme and exclusive ground truth for user-specific identity variables. The model is now explicitly directed to analyze all prior turns before drawing any conclusions about what it knows or does not know about the user. The chat history is no longer a parallel data source — it is the canonical reference for all user-attributed facts.

---

### 4.2 Elimination of Robotic AI Disclaimers

**Symptom:** When the twin encountered questions about personal habits or preferences that had not yet been shared, it defaulted to generic base-model safety responses — phrases like "As an AI language model, I do not have access to your personal schedule" — which completely shattered the persona and destroyed conversational immersion.

**Root Cause:** Standard RLHF training embeds strong reflexive responses to knowledge-boundary triggers. Unless these reflexes are explicitly overridden at the system prompt level, the model defaults to them when it detects uncertainty about personal user data.

**Resolution:** A dedicated **Data Gap Fallback Protocol** block was added to the system prompt matrix. This block explicitly forbids the use of any phrase that identifies the agent as an AI or claims an inability to access information. In its place, the model is instructed to pivot fluidly to a general biological mechanism or behavioral principle relevant to the topic at hand, and, where appropriate, to ask the user to fill in the missing detail as part of a natural conversational exchange. The gap is bridged with value, not with an apology.

---

### 4.3 Streamlit Cache Freeze on API Key Failure

**Symptom:** If the Google API key was absent, malformed, or expired at application startup, LangChain's `ChatGoogleGenerativeAI` constructor threw a Pydantic `ValidationError`. Because the initialization function was decorated with `@st.cache_resource`, Streamlit cached the exception object itself. Every subsequent page refresh — including those after the key was corrected — pulled the frozen error state directly from cache without re-executing the initialization code.

**Root Cause:** Streamlit's `@st.cache_resource` decorator is designed to cache expensive initialization objects across reruns. Its implementation treats raised exceptions as cacheable results, meaning a failed initialization is as sticky as a successful one. The session could not recover without a full application restart.

**Resolution:** The environment verification and variable injection sequence was moved to occur before the LangChain object is instantiated — outside the cached function's execution scope. The `GOOGLE_API_KEY` is validated and written to `os.environ` before the `HubermanDigitalTwin` constructor is ever called. If the key is missing, the error is raised early and explicitly, with a user-facing message, before the cache decorator has any opportunity to intercept and freeze a failed state.

---

### 4.4 Broad Identity Scope Failure in Profile Extraction

**Symptom:** The background extraction pipeline successfully captured lifestyle and behavioral data but consistently failed to log basic identity attributes. A user typing "My name is Laksh Dhawan" would see the memory panel remain blank, even though the same pipeline correctly captured dietary preferences or sleep habits mentioned in adjacent messages.

**Root Cause:** The initial extraction prompt used a narrow instruction set focused exclusively on physiological and behavioral metrics — workout splits, sleep windows, dietary restrictions, and physical deficits. Basic identity markers such as names, ages, occupations, and locations were outside the scope of what the prompt directed the model to look for, so they were systematically ignored.

**Resolution:** The extraction prompt was expanded into a **Broad Identity Extractor** that explicitly enumerates core identity indicators as a named extraction category alongside lifestyle and behavioral data. Names, ages, professional titles, geographic locations, and personal tools are now first-class extraction targets, treated with the same priority as sleep schedules or dietary patterns. The extracted output format — a single, concise, third-person bullet beginning with an asterisk — remains consistent regardless of data category, ensuring uniform rendering in the memory panel.

---

*End of Document*
