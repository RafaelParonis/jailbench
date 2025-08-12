#!/usr/bin/env python3
"""
Clean Textual TUI implementation for interactive jailbreak testing.
Built with proper documentation references and clean architecture.
"""

import asyncio
from datetime import datetime
from functools import partial
from typing import List, Optional, Dict, Any

from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.command import Provider, Hits, Hit
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.message import Message
from textual.reactive import reactive, var
from textual.screen import ModalScreen
from textual.widgets import (
    Header,
    Footer,
    Static,
    Input,
    Button,
    RichLog,
    Label,
    Pretty,
    OptionList,
    TextArea,
    LoadingIndicator,
)
from textual.widgets.option_list import Option

from .interactive_chat import InteractiveChatSession, MessageType, FlagType
from .llm_client import LiteLLMClient


class CustomTextArea(TextArea):
    """Custom TextArea that handles Ctrl+Enter for submission."""

    def check_consume_key(self, key: str, character: str | None) -> bool:
        """
        Check if we should consume a key press.

        key: str
            Key identifier.
        character: str | None
            Character associated with the key.

        Returns
        -------
        bool
            False for ctrl+enter to not consume it, True otherwise.
        """
        if key == "ctrl+enter":
            return False
        return super().check_consume_key(key, character)

    def _on_key(self, event: events.Key) -> None:
        """
        Handle key events including Ctrl+Enter for submission.

        event: events.Key
            Key event.
        """
        if event.key in ("ctrl+enter", "ctrl+j"):
            event.stop()
            self.post_message(self.Submitted())
        else:
            super()._on_key(event)

    class Submitted(Message, bubble=True):
        """
        Message posted when Ctrl+Enter is pressed.
        """
        pass


class JailbreakCommandProvider(Provider):
    """Command provider for jailbreak testing commands in the command palette."""

    async def search(self, query: str) -> Hits:
        """
        Search for jailbreak testing commands.

        query: str
            Search query from command palette.

        Yields
        ------
        Hit
            Command hits for various jailbreak commands.
        """
        matcher = self.matcher(query)
        app = self.screen.app

        if hasattr(app, 'available_clients') and len(app.available_clients) > 1:
            for client in app.available_clients:
                model_name = f"{client.provider}/{client.model}"
                command_text = f"Switch to {model_name}"
                if hasattr(app, 'current_client') and client == app.current_client:
                    command_text += " (current)"
                score = matcher.match(command_text)
                if score > 0:
                    yield Hit(
                        score,
                        matcher.highlight(command_text),
                        callable=partial(app.switch_to_model, client),
                        help=f"Switch to {model_name} model"
                    )

        commands = [
            ("Clear chat history", app.action_clear_chat, "Clear the current conversation"),
            ("Save session", app.action_save_session, "Save the current chat session"),
            ("Get jailbreak suggestions", lambda: asyncio.create_task(app.get_jailbreak_suggestions()), "Generate AI suggestions for jailbreaking"),
            ("Flag as jailbroken", lambda: app.flag_last_response("jailbroken"), "Flag last response as successfully jailbroken"),
            ("Flag as refused", lambda: app.flag_last_response("refused"), "Flag last response as refused"),
            ("Flag as partial", lambda: app.flag_last_response("partial"), "Flag last response as partially successful"),
            ("Flag as evasive", lambda: app.flag_last_response("evasive"), "Flag last response as evasive"),
            ("Flag as compliance", lambda: app.flag_last_response("compliance"), "Flag last response as compliant"),
        ]

        for command_text, callback, help_text in commands:
            score = matcher.match(command_text)
            if score > 0:
                yield Hit(
                    score,
                    matcher.highlight(command_text),
                    callable=callback,
                    help=help_text
                )


class ModelPickerScreen(ModalScreen):
    """Modal screen for selecting a model with a clean interface."""

    CSS = """
    ModelPickerScreen {
        align: center middle;
    }

    #model-picker-container {
        width: 60;
        height: auto;
        max-height: 30;
        background: $surface;
        border: thick $primary;
        padding: 1;
    }

    #model-picker-title {
        text-align: center;
        text-style: bold;
        color: $accent;
        margin-bottom: 1;
    }

    #model-list {
        min-height: 10;
        max-height: 20;
        margin: 1 0;
        border: solid $primary;
    }

    .picker-buttons {
        align: center middle;
        margin-top: 1;
    }

    .picker-buttons Button {
        margin: 0 1;
        min-width: 12;
    }
    """

    def __init__(self, clients: List[LiteLLMClient], current: LiteLLMClient):
        """
        Initialize model picker.

        clients: List[LiteLLMClient]
            Available model clients.
        current: LiteLLMClient
            Currently active client.
        """
        super().__init__()
        self.clients = clients
        self.current = current

    def compose(self) -> ComposeResult:
        """
        Compose the modal UI.

        Returns
        -------
        ComposeResult
            The composed UI for the modal.
        """
        options = []
        for i, client in enumerate(self.clients):
            is_current = client == self.current
            label = f"{client.provider}/{client.model}"
            if is_current:
                label += " ✓"
            options.append(Option(label, id=str(i)))

        with Container(id="model-picker-container"):
            yield Static("Select Model", id="model-picker-title")
            yield OptionList(*options, id="model-list")
            with Horizontal(classes="picker-buttons"):
                yield Button("Select", variant="primary", id="select")
                yield Button("Cancel", id="cancel")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """
        Handle option selection.

        event: OptionList.OptionSelected
            Selection event.
        """
        idx = int(event.option_id)
        self.dismiss(self.clients[idx])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Handle button presses.

        event: Button.Pressed
            Button press event.
        """
        if event.button.id == "select":
            option_list = self.query_one("#model-list", OptionList)
            if option_list.highlighted is not None:
                self.dismiss(self.clients[option_list.highlighted])
        else:
            self.dismiss(None)


class JailbreakTUI(App):
    """Main TUI application for interactive jailbreak testing."""

    CSS = """
    #main-container {
        layout: horizontal;
    }

    #sidebar {
        width: 30;
        border: solid $primary;
        padding: 1;
        margin-right: 1;
    }

    #chat-container {
        width: 1fr;
    }

    #conversation {
        height: 1fr;
        border: solid $primary;
        padding: 1;
        margin-bottom: 1;
    }

    #input-container {
        height: 5;
        layout: horizontal;
    }

    #message-input {
        width: 1fr;
        max-height: 5;
    }

    #send-button {
        width: 10;
        margin-left: 1;
        height: 100%;
    }

    .ai-typing {
        color: $text-muted;
        text-style: italic;
    }

    #status-bar {
        height: 2;
        background: $boost;
        padding: 0 1;
    }

    .info-section {
        margin-bottom: 1;
    }

    .info-title {
        text-style: bold;
        color: $accent;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+c", "clear_chat", "Clear Chat"),
        Binding("ctrl+s", "save_session", "Save Session"),
        Binding("ctrl+p", "command_palette", "Commands", show=True),
        Binding("ctrl+j", "send_message_now", "Send", show=False),
    ]

    current_model = reactive("No model")
    message_count = reactive(0)
    total_tokens = reactive(0)
    conversation_history = []

    def __init__(self, available_clients: List[LiteLLMClient]):
        """
        Initialize the TUI app.

        available_clients: List[LiteLLMClient]
            List of available model clients.
        """
        super().__init__()
        self.available_clients = available_clients
        self.current_client = available_clients[0] if available_clients else None
        self.session = None

    def get_command_providers(self):
        """
        Return command providers including our custom jailbreak commands.

        Returns
        -------
        set
            Set of command provider classes.
        """
        providers = super().get_command_providers()
        providers.add(JailbreakCommandProvider)
        return providers

    def compose(self) -> ComposeResult:
        """
        Compose the main UI.

        Returns
        -------
        ComposeResult
            The composed UI for the main application.
        """
        yield Header()
        with Container(id="main-container"):
            with Vertical(id="sidebar"):
                with Container(classes="info-section"):
                    yield Static("Current Model", classes="info-title")
                    yield Static(id="current-model-display")
                with Container(classes="info-section"):
                    yield Static("Session Stats", classes="info-title")
                    yield Pretty({}, id="session-stats")
                with Container(classes="info-section"):
                    yield Static("Commands", classes="info-title")
                    yield Static("Ctrl+P - Command Palette")
                    yield Static("Ctrl+C - Clear Chat")
                    yield Static("Ctrl+S - Save Session")
                    yield Static("Ctrl+Q - Quit")
            with Vertical(id="chat-container"):
                yield RichLog(id="conversation", markup=True, wrap=True)
                with Horizontal(id="input-container"):
                    yield CustomTextArea(
                        "",
                        id="message-input",
                        show_line_numbers=False,
                        soft_wrap=True,
                        tab_behavior="focus"
                    )
                    yield Button("Send\n[Ctrl+J or Ctrl+Enter]", variant="primary", id="send-button")
        yield Container(
            Static("Ready", id="status-text"),
            id="status-bar"
        )
        yield Footer()

    def on_mount(self) -> None:
        """
        Initialize when app mounts.
        """
        if self.current_client:
            self.session = InteractiveChatSession(self.current_client)
            self.session.start_new_session()
            self.update_model_display()
            self.update_stats()
        self.query_one("#message-input", CustomTextArea).focus()

    def on_custom_text_area_submitted(self, message: CustomTextArea.Submitted) -> None:
        """
        Handle CustomTextArea submission via Ctrl+Enter.

        message: CustomTextArea.Submitted
            Submission message.
        """
        self.submit_message()

    def submit_message(self) -> None:
        """
        Submit the message from TextArea.
        """
        text_area = self.query_one("#message-input", CustomTextArea)
        message = text_area.text.strip()
        if not message:
            return
        text_area.clear()
        if message.startswith('/'):
            self.handle_command(message)
        else:
            asyncio.create_task(self.send_message(message))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """
        Handle button presses.

        event: Button.Pressed
            Button press event.
        """
        if event.button.id == "send-button":
            self.submit_message()

    def action_send_message_now(self) -> None:
        """
        Action invoked by key binding to submit current TextArea text.
        Only triggers if the message input has focus, to avoid accidental sends.
        """
        focused = self.focused
        if isinstance(focused, CustomTextArea) and focused.id == "message-input":
            self.submit_message()

    def handle_command(self, command: str) -> None:
        """
        Handle slash commands.

        command: str
            Command string starting with /
        """
        parts = command.split(' ', 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        if cmd == "/help":
            self.show_help()
        elif cmd == "/model":
            self.show_model_picker()
        elif cmd == "/clear":
            self.action_clear_chat()
        elif cmd == "/save":
            self.action_save_session()
        elif cmd == "/system":
            if args:
                self.set_system_prompt(args)
            else:
                self.update_status("Usage: /system <prompt>")
        elif cmd == "/flag":
            if args:
                self.flag_last_response(args)
            else:
                self.update_status("Usage: /flag <jailbroken|refused|partial|evasive|compliance>")
        elif cmd == "/note":
            if args:
                self.add_note(args)
            else:
                self.update_status("Usage: /note <your note>")
        elif cmd == "/suggest":
            asyncio.create_task(self.get_jailbreak_suggestions())
        else:
            self.update_status(f"Unknown command: {cmd}")

    async def send_message(self, message: str) -> None:
        """
        Send message to the AI model with typing animation.

        message: str
            User message.
        """
        conversation = self.query_one("#conversation", RichLog)
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = message.replace('\n', '\n    ')
        user_display = f"[dim]{timestamp}[/dim] [bold green]You:[/bold green]\n    {formatted_message}"
        conversation.write(user_display)
        self.conversation_history.append(('user', timestamp, message))
        ai_timestamp = datetime.now().strftime("%H:%M:%S")
        model_name = f"{self.current_client.provider}/{self.current_client.model}"
        typing_display = f"[dim]{ai_timestamp}[/dim] [bold blue]{model_name}:[/bold blue]\n    [dim italic]typing...[/dim italic]"
        conversation.write(typing_display)
        conversation.scroll_end()
        self.update_status("Generating response...")
        try:
            response, response_time, analysis = await asyncio.get_event_loop().run_in_executor(
                None,
                self.session.send_user_message,
                message
            )
            conversation.clear()
            for entry_type, entry_time, entry_content in self.conversation_history:
                if entry_type == 'user':
                    formatted = entry_content.replace('\n', '\n    ')
                    conversation.write(f"[dim]{entry_time}[/dim] [bold green]You:[/bold green]\n    {formatted}")
                elif entry_type == 'assistant':
                    formatted = entry_content.replace('\n', '\n    ')
                    conversation.write(f"[dim]{entry_time}[/dim] [bold blue]{model_name}:[/bold blue]\n    {formatted}")
                elif entry_type == 'system':
                    conversation.write(f"[dim]{entry_time}[/dim] [bold yellow]SYSTEM:[/bold yellow] {entry_content}")
                elif entry_type == 'flag':
                    conversation.write(f"[dim]{entry_time}[/dim] [bold magenta]FLAG:[/bold magenta] {entry_content}")
                elif entry_type == 'note':
                    conversation.write(f"[dim]{entry_time}[/dim] [bold cyan]NOTE:[/bold cyan] {entry_content}")
            formatted_response = response.replace('\n', '\n    ')
            conversation.write(f"[dim]{ai_timestamp}[/dim] [bold blue]{model_name}:[/bold blue]\n    {formatted_response}")
            self.conversation_history.append(('assistant', ai_timestamp, response))
            self.message_count += 1
            if analysis and 'usage' in analysis:
                self.total_tokens += analysis['usage'].get('total_tokens', 0)
            self.update_stats()
            self.update_status(f"Response received in {response_time:.2f}s")
            conversation.scroll_end()
        except Exception as e:
            conversation.clear()
            for entry_type, entry_time, entry_content in self.conversation_history:
                if entry_type == 'user':
                    formatted = entry_content.replace('\n', '\n    ')
                    conversation.write(f"[dim]{entry_time}[/dim] [bold green]You:[/bold green]\n    {formatted}")
                elif entry_type == 'assistant':
                    formatted = entry_content.replace('\n', '\n    ')
                    conversation.write(f"[dim]{entry_time}[/dim] [bold blue]{model_name}:[/bold blue]\n    {formatted}")
                elif entry_type == 'system':
                    conversation.write(f"[dim]{entry_time}[/dim] [bold yellow]SYSTEM:[/bold yellow] {entry_content}")
            conversation.write(f"[bold red]Error:[/bold red] {str(e)}")
            self.update_status(f"Error: {str(e)}")

    def show_model_picker(self) -> None:
        """
        Show the model picker modal.
        """
        if len(self.available_clients) <= 1:
            self.update_status("Only one model available")
            return

        def handle_selection(selected_client: Optional[LiteLLMClient]) -> None:
            """
            Handle model selection.

            selected_client: Optional[LiteLLMClient]
                Selected client or None.
            """
            if selected_client and selected_client != self.current_client:
                self.switch_to_model(selected_client)

        self.push_screen(
            ModelPickerScreen(self.available_clients, self.current_client),
            handle_selection
        )

    def switch_to_model(self, new_client: LiteLLMClient) -> None:
        """
        Switch to a different model.

        new_client: LiteLLMClient
            New model client to switch to.
        """
        if new_client == self.current_client:
            return
        old_model = f"{self.current_client.provider}/{self.current_client.model}"
        new_model = f"{new_client.provider}/{new_client.model}"
        if self.session and self.session.current_session:
            self.session.save_session()
        self.current_client = new_client
        self.session = InteractiveChatSession(new_client)
        self.session.start_new_session()
        self.update_model_display()
        self.update_stats()
        conversation = self.query_one("#conversation", RichLog)
        timestamp = datetime.now().strftime("%H:%M:%S")
        conversation.write(
            f"[dim]{timestamp}[/dim] [bold yellow]SYSTEM:[/bold yellow] Switched from {old_model} to {new_model}"
        )
        self.update_status(f"Switched to {new_model}")

    def action_clear_chat(self) -> None:
        """
        Clear the chat conversation.
        """
        if self.session and self.session.current_session:
            self.session.save_session()
        self.session.start_new_session()
        self.query_one("#conversation", RichLog).clear()
        self.message_count = 0
        self.total_tokens = 0
        self.update_stats()
        self.update_status("Chat cleared")

    def action_save_session(self) -> None:
        """
        Save the current session.
        """
        if self.session:
            self.session.save_session()
            self.update_status("Session saved")

    def set_system_prompt(self, prompt: str) -> None:
        """
        Set a new system prompt.

        prompt: str
            New system prompt.
        """
        self.session.start_new_session(prompt)
        self.query_one("#conversation", RichLog).clear()
        conversation = self.query_one("#conversation", RichLog)
        timestamp = datetime.now().strftime("%H:%M:%S")
        conversation.write(
            f"[dim]{timestamp}[/dim] [bold yellow]SYSTEM:[/bold yellow] New session with custom prompt"
        )
        self.update_status("System prompt updated")

    def flag_last_response(self, flag_type: str) -> None:
        """
        Flag the last AI response.

        flag_type: str
            Type of flag to apply.
        """
        flag_mapping = {
            'jailbroken': FlagType.JAILBROKEN,
            'almost': FlagType.ALMOST_JAILBROKEN,
            'refused': FlagType.REFUSED,
            'partial': FlagType.PARTIAL_SUCCESS,
            'evasive': FlagType.EVASIVE,
            'compliance': FlagType.COMPLIANCE
        }
        flag_type_lower = flag_type.lower()
        if flag_type_lower in flag_mapping:
            if self.session:
                self.session.flag_message(flag_mapping[flag_type_lower])
                conversation = self.query_one("#conversation", RichLog)
                timestamp = datetime.now().strftime("%H:%M:%S")
                conversation.write(
                    f"[dim]{timestamp}[/dim] [bold magenta]FLAG:[/bold magenta] Response flagged as {flag_type_lower}"
                )
                self.update_status(f"Response flagged as {flag_type_lower}")
        else:
            self.update_status("Invalid flag type. Use: jailbroken, refused, partial, evasive, or compliance")

    def add_note(self, note: str) -> None:
        """
        Add a note to the conversation.

        note: str
            Note text to add.
        """
        if self.session:
            self.session.add_note(note)
            conversation = self.query_one("#conversation", RichLog)
            timestamp = datetime.now().strftime("%H:%M:%S")
            conversation.write(
                f"[dim]{timestamp}[/dim] [bold cyan]NOTE:[/bold cyan] {note}"
            )
            self.update_status("Note added to conversation")

    async def get_jailbreak_suggestions(self) -> None:
        """
        Get AI-generated jailbreak suggestions.
        """
        conversation = self.query_one("#conversation", RichLog)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.update_status("Generating jailbreak suggestions...")
        try:
            if self.session:
                suggestions = self.session.get_jailbreak_suggestions()
                conversation.write(
                    f"[dim]{timestamp}[/dim] [bold yellow]AI SUGGESTIONS:[/bold yellow]"
                )
                for msg in reversed(self.session.current_session.messages):
                    if msg.message_type == MessageType.SUGGESTION:
                        conversation.write(msg.content)
                        break
                self.update_status("Suggestions generated")
        except Exception as e:
            conversation.write(f"[bold red]Error:[/bold red] {str(e)}")
            self.update_status(f"Error generating suggestions: {str(e)}")

    def show_help(self) -> None:
        """
        Show help information.
        """
        help_text = """[bold]Available Commands:[/bold]
        
/help - Show this help message
/model - Switch to a different model
/clear - Clear conversation history
/save - Save current session
/system <prompt> - Set custom system prompt
/flag <type> - Flag last response (jailbroken|refused|partial|evasive|compliance)
/note <text> - Add a note to conversation
/suggest - Get AI-generated jailbreak suggestions
        
[bold]Keyboard Shortcuts:[/bold]
Ctrl+P - Open command palette
Ctrl+C - Clear chat
Ctrl+S - Save session
Ctrl+Q - Quit"""
        conversation = self.query_one("#conversation", RichLog)
        timestamp = datetime.now().strftime("%H:%M:%S")
        conversation.write(f"[dim]{timestamp}[/dim] [bold yellow]HELP[/bold yellow]")
        conversation.write(help_text)

    def update_status(self, message: str) -> None:
        """
        Update status bar.

        message: str
            Status message.
        """
        self.query_one("#status-text", Static).update(message)

    def update_model_display(self) -> None:
        """
        Update current model display.
        """
        if self.current_client:
            model_text = f"{self.current_client.provider}/{self.current_client.model}"
            self.query_one("#current-model-display", Static).update(model_text)

    def update_stats(self) -> None:
        """
        Update session statistics display.
        """
        stats = {
            "Messages": self.message_count,
            "Total Tokens": self.total_tokens,
        }
        if self.session and self.session.current_session:
            stats["Session ID"] = self.session.current_session.session_id[:8]
        self.query_one("#session-stats", Pretty).update(stats)