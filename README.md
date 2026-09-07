# Oswald Conversation for Home Assistant

Use Oswald AI as a Home Assistant conversation agent.

Requires Home Assistant **2026.7.0 or newer** and Oswald's authenticated Home
Assistant gateway, **protocol v1**. The older generic WebSocket gateway is not
supported. The integration is text-only; tools run on Oswald, not in this client.

## Installation

### Via [HACS](https://hacs.xyz/)

[![Open your Home Assistant instance and open a repository inside the Home Assistant Community Store.](https://my.home-assistant.io/badges/hacs_repository.svg)](https://my.home-assistant.io/redirect/hacs_repository/?owner=jonahgcarpenter&repository=has-oswald-conversation&category=integration)

## Configuration

Enable the gateway on the Oswald host with both `HOME_ASSISTANT_LISTEN_PORT`
(for example, `8000`) and `HOME_ASSISTANT_AUTH_TOKEN` (a randomly generated secret
of at least 32 bytes). Restart Oswald after changing these settings. Enter that
same token in Home Assistant; it is not a Home Assistant access token.

The endpoint is `/homeassistant/ws`, for example
`wss://oswald.example/homeassistant/ws`. Oswald's listener does not provide TLS;
use a TLS-terminating reverse proxy for `wss://`. Preserve the `Authorization`
header and do not add an `Origin` header. Use `wss://` outside a trusted local
connection: `ws://` transmits the bearer token and conversation without encryption.
Do not expose the gateway directly to the internet.

The default `ws://127.0.0.1:8000/homeassistant/ws` points to **Home Assistant's own
runtime/container**, not necessarily the machine running Oswald. When they run
separately, use a gateway address reachable from Home Assistant.

1. Go to Settings -> Devices & services.
2. Add Integration.
3. Search for Oswald Conversation.
4. Enter the Oswald Home Assistant WebSocket URL and authentication token.
5. Optionally select a default user for requests without an authenticated Home Assistant user.

Multiple integration entries are supported, including entries for the same
gateway with different default users. Each creates a separate conversation agent;
select the intended agent in each Assist pipeline. Reconfigure and
reauthentication apply only to the selected entry.

### Default User Identity

The default user applies to **all unauthenticated requests**, including voice
satellites, not just a particular device. They share that user's Oswald identity,
memory, linked accounts, and tool access. Only enable this fallback if you trust
every device that can access the Assist pipeline. Prefer a dedicated
least-privileged account; do not link it to a privileged Oswald account. The
integration rejects deleted, inactive, and system-generated fallback users.
Leave the field empty to require an authenticated user, or clear it in
Reconfigure to disable the fallback. Authenticated requests use their own user.

### Reconfigure And Upgrade

Use the integration's **Reconfigure** menu to change the URL, token, or default
user. Leaving the token blank or unchanged at its masked value preserves an
existing token. Entries without a stored token require a new one.

Older tokenless installations prompt for authentication when loaded. A rejected
token during a conversation also starts Home Assistant's reauthentication flow.
Enter the gateway token to recover the existing entry without deleting it. If
upgrading from the generic gateway, first update the endpoint via Reconfigure.
To rotate credentials, change the token on Oswald, restart it, and supply the new
token through Reconfigure or reauthentication.

## Setup

1. Go to Settings -> Voice Assistants.
2. Add assistant or edit existing.
3. Change conversation agent to Oswald.

## Conversations And Cancellation

Each turn opens a new connection while retaining the Home Assistant conversation
ID. Reasoning and external tool activity update the conversation timeline live.
**Answer text is published only when Oswald sends its terminal response.** HA's
text stream is append-only, and Oswald can correct its earlier previews; waiting
for the final answer prevents speech from reading stale, incomplete, or duplicate
answers. This trades token-by-token answer latency for correctness.

Connection setup, readiness, and request submission have a ten-second combined
deadline. After submission there is no integration-imposed total or idle timeout;
long model calls and multiple tool rounds can take more than four minutes. HA's
Assist pipeline, proxies, and model providers may impose their own limits.

Disconnecting or canceling the local HA task **does not stop Oswald's work**.
Interrupted submitted requests report an unknown outcome and are never retried
automatically: actions may already have completed. To cancel active work, send
`/stop` as another turn in the **same conversation and with the same user**. This
does not remove queued requests or undo completed actions. Check the outcome
before repeating an interrupted request.

Text command exports such as `/memories list` are returned inline. Binary/image
attachments are unsupported. Reasoning and tool details can appear in HA chat
history and debug logs; treat those records as private conversation data.

## Development

Run the tests with Python 3.14 in an isolated environment:

```sh
python -m pip install -r requirements-test.txt
python -m pytest -q
```

For the minimum supported HA release, use a separate environment with
`requirements-test-min.txt`. The suites use real HA config-flow and ChatLog APIs
with a mocked gateway; network access is disabled during tests. CI tests HA
2026.7.0 and 2026.9.1 and runs lint, formatting, JSON, Hassfest, and HACS validation
on pull requests and pushes. Update the pinned HA test dependencies together
with their conversation requirements when advancing the current-release target.
