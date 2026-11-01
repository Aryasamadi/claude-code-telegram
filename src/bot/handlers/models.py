"""Telegram commands for registering and selecting custom model providers."""
from __future__ import annotations

import html
import shlex

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes

from src.providers.service import ModelRoute, ProviderService

PAGE_SIZE = 8


def _service(context: ContextTypes.DEFAULT_TYPE) -> ProviderService | None:
    service = context.bot_data.get("provider_service")
    return service if isinstance(service, ProviderService) else None


def _page_markup(
    models: list[ModelRoute], page: int, current_alias: str | None
) -> tuple[str, InlineKeyboardMarkup]:
    total_pages = max(1, (len(models) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = max(0, min(page, total_pages - 1))
    start = page * PAGE_SIZE
    visible = models[start : start + PAGE_SIZE]

    lines = [
        "<b>Available models</b>",
        f"Page {page + 1}/{total_pages} · {len(models)} models",
        "",
    ]
    keyboard: list[list[InlineKeyboardButton]] = []
    for route in visible:
        selected = route.alias == current_alias
        marker = "✅ " if selected else ""
        lines.append(
            f"{marker}<b>{html.escape(route.provider_name)}</b> — "
            f"<code>{html.escape(route.model_id)}</code>"
        )
        label = f"{marker}{route.provider_name}: {route.model_id}"
        if len(label) > 62:
            label = label[:59] + "..."
        keyboard.append(
            [InlineKeyboardButton(label, callback_data=f"mdl:s:{route.alias}")]
        )

    nav: list[InlineKeyboardButton] = []
    if page > 0:
        nav.append(
            InlineKeyboardButton("‹ Previous", callback_data=f"mdl:p:{page - 1}")
        )
    if page + 1 < total_pages:
        nav.append(
            InlineKeyboardButton("Next ›", callback_data=f"mdl:p:{page + 1}")
        )
    if nav:
        keyboard.append(nav)
    return "\n".join(lines), InlineKeyboardMarkup(keyboard)


async def add_model_provider(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Add a provider from: /addmodel "name" "base_url" "model" "api_key"."""
    message = update.effective_message
    if message is None:
        return

    raw_text = message.text or ""
    # The API key is part of the command, so remove the source message before
    # doing validation/network calls. Never log or echo raw command arguments.
    deleted = True
    try:
        await message.delete()
    except Exception:
        deleted = False

    try:
        parts = shlex.split(raw_text)
    except ValueError:
        await update.effective_chat.send_message(
            "Invalid command quoting. Use four quoted values after /addmodel."
        )
        return

    if len(parts) != 5 or parts[0].split("@", 1)[0].lower() != "/addmodel":
        await update.effective_chat.send_message(
            'Usage:\n<code>/addmodel "Provider Name" "https://host/v1" "model-id" "API_KEY"</code>',
            parse_mode="HTML",
        )
        return

    service = _service(context)
    if service is None:
        await update.effective_chat.send_message("Provider manager is not available.")
        return

    if not deleted:
        await update.effective_chat.send_message(
            "⚠️ I could not delete the command containing your API key. "
            "Please delete that message manually."
        )

    _, provider_name, base_url, model_id, api_key = parts
    try:
        count, catalog_available = await service.add_provider(
            provider_name, base_url, model_id, api_key
        )
        settings = context.bot_data["settings"]
        selected = await service.select_model(
            next(
                route.alias
                for route in await service.list_models()
                if route.provider_name == provider_name
                and route.model_id == model_id
                and route.base_url.rstrip("/") == base_url.rstrip("/")
            )
        )
        # The requested model is the initial active model; other discovered
        # models remain available from /model and /models.
        status = "Provider registered and proxy is ready."
        if not catalog_available:
            status += (
                "\nThe provider's /models endpoint was unavailable; "
                "only the model from your command was added."
            )
        await update.effective_chat.send_message(
            f"✅ <b>{html.escape(provider_name)}</b> added.\n"
            f"Models found: <b>{count}</b>\n"
            f"Active model: <code>{html.escape(selected.model_id if selected else model_id)}</code>\n\n"
            f"{status}\nUse /model to switch models or /models to browse them.",
            parse_mode="HTML",
        )
    except Exception as exc:
        # Errors are intentionally generic: provider URLs and API keys must not
        # leak back into Telegram messages or logs.
        await update.effective_chat.send_message(
            f"❌ Could not add this provider: {html.escape(str(exc))}",
            parse_mode="HTML",
        )


async def show_models(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the provider/model picker, similar to Hermes /model."""
    message = update.effective_message
    if message is None:
        return
    service = _service(context)
    if service is None:
        await update.effective_chat.send_message("Provider manager is not available.")
        return
    models = await service.list_models()
    if not models:
        await update.effective_chat.send_message(
            'No providers added yet.\nAdd one with:\n'
            '<code>/addmodel "Provider Name" "https://host/v1" "model-id" "API_KEY"</code>',
            parse_mode="HTML",
        )
        return
    settings = context.bot_data["settings"]
    body, keyboard = _page_markup(models, 0, settings.claude_model)
    await update.effective_chat.send_message(body, parse_mode="HTML", reply_markup=keyboard)


async def model_picker_callback(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    """Handle pagination and model-selection buttons."""
    query = update.callback_query
    if query is None:
        return
    await query.answer()
    service = _service(context)
    if service is None:
        await query.edit_message_text("Provider manager is not available.")
        return

    data = query.data or ""
    settings = context.bot_data["settings"]
    if data.startswith("mdl:s:"):
        alias = data.removeprefix("mdl:s:")
        route = await service.select_model(alias)
        if route is None:
            await query.edit_message_text("This model is no longer available.")
            return
        await query.edit_message_text(
            f"✅ Active model changed to:\n"
            f"<b>{html.escape(route.provider_name)}</b>\n"
            f"<code>{html.escape(route.model_id)}</code>",
            parse_mode="HTML",
        )
        return

    page = 0
    if data.startswith("mdl:p:"):
        try:
            page = int(data.removeprefix("mdl:p:"))
        except ValueError:
            page = 0
    models = await service.list_models()
    if not models:
        await query.edit_message_text("No models are registered.")
        return
    body, keyboard = _page_markup(models, page, settings.claude_model)
    await query.edit_message_text(body, parse_mode="HTML", reply_markup=keyboard)
