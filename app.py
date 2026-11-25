import os
import json
import html
import time
import asyncio
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import requests
import pandas as pd
import altair as alt
import streamlit as st
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    function_tool,
    set_tracing_disabled,
)

load_dotenv(Path(__file__).resolve().parent / ".env")
set_tracing_disabled(True)  # no OpenAI tracing key needed

APP_NAME = "Breeze"

st.set_page_config(
    page_title=f"{APP_NAME} · Live weather",
    page_icon="🌤️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# Config: choose the AI with LLM_PROVIDER=groq | gemini | claude in .env
# ------------------------------------------------------------------
PROVIDERS = {
    "groq": {
        "label": "Groq",
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model_env": "GROQ_MODEL",
        "default_model": "openai/gpt-oss-120b",
    },
    "gemini": {
        "label": "Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model_env": "GEMINI_MODEL",
        "default_model": "gemini-2.5-flash",
    },
    "claude": {
        "label": "Claude",
        "base_url": "https://api.anthropic.com/v1/",
        "key_env": "ANTHROPIC_API_KEY",
        "model_env": "CLAUDE_MODEL",
        "default_model": "claude-haiku-5-5",
    },
}

PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
if PROVIDER not in PROVIDERS:
    PROVIDER = "groq"
CFG = PROVIDERS[PROVIDER]
LLM_KEY = os.getenv(CFG["key_env"], "").strip()
MODEL = os.getenv(CFG["model_env"], CFG["default_model"]).strip()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "").strip()

# ------------------------------------------------------------------
# State (set early: the page background depends on it)
# ------------------------------------------------------------------
st.session_state.setdefault("messages", [])
st.session_state.setdefault("agent_history", [])
st.session_state.setdefault("pending", None)
st.session_state.setdefault("bg", "default")

# ------------------------------------------------------------------
# Weather backgrounds (applied when the user asks about overall weather)
# ------------------------------------------------------------------
BG = {
    "default": {
        "bg": "radial-gradient(900px 520px at 6% -8%, #FFE8D2 0%, transparent 60%), "
              "radial-gradient(820px 500px at 100% 0%, #D8EBFF 0%, transparent 58%), #F6F9FC",
    },
    "clear_day": {
        "bg": "radial-gradient(circle at 86% 7%, #FFEFB0 0, #FFD76A 60px, rgba(255,215,106,.35) 150px, transparent 300px), "
              "linear-gradient(180deg, #A9DAFF 0%, #DDF0FF 55%, #F7FBFF 100%)",
    },
    "clear_night": {
        "bg": "radial-gradient(circle at 86% 7%, #FFF6D0 0, #FFF6D0 26px, rgba(255,246,208,.3) 70px, transparent 160px), "
              "linear-gradient(180deg, #8EAEE0 0%, #C3D5F1 60%, #E8EFFA 100%)",
    },
    "clouds": {
        "bg": "radial-gradient(ellipse 280px 90px at 18% 10%, rgba(255,255,255,.95), transparent 70%), "
              "radial-gradient(ellipse 340px 100px at 78% 20%, rgba(255,255,255,.85), transparent 70%), "
              "linear-gradient(180deg, #BCCFE0 0%, #DFE8F0 70%, #F3F6F9 100%)",
    },
    "rain": {
        "bg": "linear-gradient(105deg, transparent 48%, rgba(255,255,255,.55) 50%, transparent 52%), "
              "linear-gradient(180deg, #A5B9CC 0%, #C9D6E2 70%, #E4EBF1 100%)",
        "size": "42px 84px, auto",
        "anim": "rainfall 0.8s linear infinite",
    },
    "storm": {
        "bg": "linear-gradient(105deg, transparent 48%, rgba(255,255,255,.5) 50%, transparent 52%), "
              "linear-gradient(180deg, #95A0BD 0%, #B8BFD3 70%, #DCE0EB 100%)",
        "size": "42px 84px, auto",
        "anim": "rainfall 0.5s linear infinite",
    },
    "snow": {
        "bg": "radial-gradient(3px 3px at 20px 30px, #fff, transparent), "
              "radial-gradient(2px 2px at 70px 90px, #fff, transparent), "
              "radial-gradient(4px 4px at 120px 50px, #fff, transparent), "
              "linear-gradient(180deg, #D3E3F2 0%, #ECF3F9 70%, #FFFFFF 100%)",
        "size": "160px 160px, 160px 160px, 160px 160px, auto",
        "anim": "snowfall 6s linear infinite",
    },
    "mist": {"bg": "linear-gradient(180deg, #CBD3DA 0%, #E5E9ED 70%, #F5F6F7 100%)"},
}


def bg_key(main: str, icon: str) -> str:
    if main == "Clear":
        return "clear_night" if str(icon).endswith("n") else "clear_day"
    return {
        "Clouds": "clouds", "Rain": "rain", "Drizzle": "rain", "Thunderstorm": "storm",
        "Snow": "snow", "Mist": "mist", "Fog": "mist", "Haze": "mist",
        "Smoke": "mist", "Dust": "mist", "Sand": "mist",
    }.get(main, "default")


def bg_css(key: str) -> str:
    s = BG.get(key, BG["default"])
    css = f".stApp {{ background: {s['bg']}; background-attachment: fixed;"
    if s.get("size"):
        css += f" background-size: {s['size']};"
    if s.get("anim"):
        css += f" animation: {s['anim']};"
    return css + " }"


# ------------------------------------------------------------------
# Design system: Instrument Serif (display) + Plus Jakarta Sans (UI)
# Glass surfaces, soft depth, one blue accent, consistent 8px rhythm.
# ------------------------------------------------------------------
BASE_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

:root {
    color-scheme: light;
    --surface: rgba(255,255,255,.86);
    --solid: #FFFFFF;
    --surface-2: #F1F6FA;
    --ink: #0B1F33;
    --ink-2: #34495E;
    --muted: #5B7083;
    --line: rgba(11,31,51,.08);
    --accent: #2563EB;
    --accent-2: #7C3AED;
    --accent-soft: #E7EFFE;
    --shadow-sm: 0 1px 2px rgba(11,31,51,.05), 0 4px 10px -4px rgba(11,31,51,.08);
    --shadow: 0 1px 2px rgba(11,31,51,.04), 0 18px 36px -16px rgba(11,31,51,.22);
    --serif: 'Instrument Serif', Georgia, serif;
    --sans: 'Plus Jakarta Sans', system-ui, sans-serif;
    --r-lg: 26px; --r-md: 18px; --r-sm: 12px;
}
html, body, [class*="css"], .stApp { font-family: var(--sans); }
.stApp { color: var(--ink); }
::selection { background: var(--accent-soft); }

@keyframes rainfall { from { background-position: 0 0, 0 0; } to { background-position: 0 84px, 0 0; } }
@keyframes snowfall { from { background-position: 0 0, 0 0, 0 0, 0 0; } to { background-position: 0 160px, 0 160px, 0 160px, 0 0; } }
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
@media (prefers-reduced-motion: reduce) { .stApp { animation: none !important; } * { transition: none !important; animation: none !important; } }

#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] { display: none; }
.block-container { max-width: 820px; padding: 1.25rem 1.25rem 8rem 1.25rem; }

/* Top bar */
.topbar {
    display: flex; align-items: center; justify-content: space-between; gap: .75rem;
    background: var(--surface); backdrop-filter: blur(16px) saturate(140%); -webkit-backdrop-filter: blur(16px) saturate(140%);
    border: 1px solid var(--line); border-radius: 999px;
    padding: .5rem .6rem .5rem 1rem; box-shadow: var(--shadow);
}
.logo { display: flex; align-items: center; gap: .6rem; min-width: 0; }
.logo-dot {
    flex: none; width: 1.55rem; height: 1.55rem; border-radius: 50%;
    background: radial-gradient(circle at 32% 30%, #FFE9A8, #FFB347 62%, #FF9A5C);
    box-shadow: 0 0 0 4px rgba(255,179,71,.22);
}
.logo-word { font-family: var(--serif); font-size: 1.7rem; line-height: 1; letter-spacing: -0.01em; }
.engine {
    display: inline-flex; align-items: center; gap: .45rem;
    font-size: .74rem; font-weight: 600; color: var(--ink-2);
    background: var(--surface-2); border-radius: 999px; padding: .4rem .85rem;
    max-width: 52vw; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.engine::before { content: ""; width: .5rem; height: .5rem; border-radius: 50%; background: #22B573; box-shadow: 0 0 0 3px rgba(34,181,115,.2); flex: none; }

/* Hero */
.hero { position: relative; padding: 2.6rem .25rem 1.2rem .25rem; animation: rise .5s ease both; }
.hero::before {
    content: ""; position: absolute; right: 2%; top: .6rem; width: 150px; height: 150px; border-radius: 50%;
    background: radial-gradient(circle at 35% 35%, #FFE3A3 0%, #FFC4A3 55%, rgba(255,196,163,0) 72%);
    opacity: .85; z-index: 0; pointer-events: none;
}
.hero > * { position: relative; z-index: 1; }
.eyebrow { font-size: .72rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
.hero-title { font-family: var(--serif); font-size: clamp(2.6rem, 8.5vw, 4.4rem); line-height: 1; letter-spacing: -0.02em; margin: .7rem 0 .9rem 0; }
.hero-title em { background: linear-gradient(90deg, var(--accent), var(--accent-2)); -webkit-background-clip: text; background-clip: text; color: transparent; font-style: italic; padding-right: .08em; }
.hero-sub { color: var(--ink-2); font-size: clamp(.95rem, 2.8vw, 1.08rem); line-height: 1.6; max-width: 34rem; margin: 0; }
.try { margin: 1.4rem 0 .6rem 0; }

/* Action bar (units + new chat + export) */
.st-key-actionbar { margin: 1rem 0 .4rem 0; }
.st-key-actionbar [data-testid="stHorizontalBlock"] { align-items: center; gap: .6rem; }

/* Chat */
[data-testid="stChatMessage"] { background: transparent; border: none; padding: .35rem 0; margin-bottom: .25rem; animation: rise .35s ease both; }
[data-testid="stChatMessageAvatarUser"], [data-testid="stChatMessageAvatarAssistant"], [data-testid*="chatAvatarIcon"] { display: none; }
[data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li { color: var(--ink); line-height: 1.7; font-size: 1rem; }
[data-testid="stChatMessage"]:has([data-testid*="AvatarUser"]) { flex-direction: row-reverse; }
[data-testid="stChatMessage"]:has([data-testid*="AvatarUser"]) [data-testid="stChatMessageContent"] {
    width: fit-content; max-width: 86%; margin-left: auto;
    background: linear-gradient(135deg, #0B1F33, #1D3A5C); border-radius: 20px 20px 6px 20px; padding: .7rem 1.1rem;
    box-shadow: var(--shadow-sm);
}
[data-testid="stChatMessage"]:has([data-testid*="AvatarUser"]) p { color: #FFFFFF; margin: 0; }
[data-testid="stChatMessage"]:has([data-testid*="AvatarAssistant"]) [data-testid="stChatMessageContent"] > div:last-child p { color: var(--ink-2); }

/* Status pill */
[data-testid="stStatus"] details, [data-testid="stStatusWidget"] { border-radius: var(--r-md); border-color: var(--line); background: var(--surface); box-shadow: var(--shadow-sm); }

/* Input */
[data-testid="stBottom"], [data-testid="stBottom"] > div { background: transparent; }
[data-testid="stChatInput"] > div {
    background: var(--surface); backdrop-filter: blur(14px); border: 1px solid var(--line); border-radius: 22px; box-shadow: var(--shadow);
}
[data-testid="stChatInput"] > div:focus-within { border-color: var(--accent); box-shadow: 0 0 0 4px rgba(37,99,235,.14); }
[data-testid="stChatInput"] textarea { color: var(--ink); font-family: var(--sans); }
[data-testid="stChatInput"] textarea::placeholder { color: var(--muted); }
[data-testid="stChatInput"] button { background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: #fff; border-radius: 14px; }

/* Buttons */
.stButton > button, .stDownloadButton > button {
    width: 100%; background: var(--solid); color: var(--ink);
    border: 1px solid var(--line); border-radius: 16px; padding: .85rem 1rem; font-weight: 600;
    justify-content: flex-start; text-align: left; box-shadow: var(--shadow-sm);
    transition: transform .15s, border-color .15s, background .15s, color .15s, box-shadow .15s;
}
.stButton > button:hover, .stDownloadButton > button:hover { border-color: var(--accent); color: var(--accent); transform: translateY(-1px); box-shadow: var(--shadow); }
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible { outline: 3px solid rgba(37,99,235,.35); outline-offset: 2px; }
.stButton > button:disabled, .stDownloadButton > button:disabled { opacity: .45; transform: none; box-shadow: none; cursor: not-allowed; }

/* Pill action buttons */
.st-key-new_chat button, .st-key-export_chat button {
    padding: .55rem 1rem; justify-content: center; border-radius: 999px; font-size: .85rem; text-align: center;
}
.st-key-new_chat button { background: linear-gradient(135deg, var(--accent), var(--accent-2)); color: #fff; border: none; }
.st-key-new_chat button:hover { color: #fff; filter: brightness(1.08); }
.st-key-new_chat button:disabled { background: var(--solid); color: var(--muted); border: 1px solid var(--line); }

[data-testid="stSegmentedControl"] { background: var(--solid); border: 1px solid var(--line); border-radius: 999px; padding: 3px; width: fit-content; box-shadow: var(--shadow-sm); }
[data-testid="stSegmentedControl"] button { font-family: var(--sans); font-weight: 700; border-radius: 999px; }
[data-testid="stRadio"] > div { background: var(--solid); padding: .25rem .9rem; border-radius: 999px; width: fit-content; border: 1px solid var(--line); }
[data-testid="stRadio"] label p { color: var(--ink); font-weight: 600; }
[data-testid="stExpander"] details { border-radius: 16px; border-color: var(--line); background: var(--solid); }

/* Cards */
.card { background: var(--surface); backdrop-filter: blur(12px); border: 1px solid var(--line); border-radius: var(--r-lg); padding: 1.3rem 1.35rem; box-shadow: var(--shadow); margin-bottom: .9rem; animation: rise .45s ease both; }
.card-head { display: flex; justify-content: space-between; align-items: flex-start; gap: .8rem; flex-wrap: wrap; margin-bottom: 1rem; }
.card-title { font-family: var(--serif); font-size: 1.7rem; line-height: 1.1; margin-top: .2rem; }

/* Chart container */
.chart-card { background: var(--surface); backdrop-filter: blur(12px); border: 1px solid var(--line); border-radius: var(--r-lg); padding: 1.3rem 1.35rem .6rem 1.35rem; box-shadow: var(--shadow); margin-bottom: .9rem; animation: rise .45s ease both; }
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] { background: transparent; border: none; padding: 0; box-shadow: none; margin-top: -.4rem; margin-bottom: .9rem; }
.chart-note { display: flex; gap: 1rem; flex-wrap: wrap; font-size: .75rem; color: var(--muted); font-weight: 600; margin: -.2rem 0 1rem 0; }
.dot { display: inline-block; width: .55rem; height: .55rem; border-radius: 50%; margin-right: .4rem; vertical-align: middle; }

/* Stat strip */
.strip {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(104px, 1fr)); gap: 1px;
    background: rgba(11,31,51,.09); border-radius: 16px; overflow: hidden;
}
.stat { background: rgba(255,255,255,.78); padding: .7rem .85rem; display: flex; flex-direction: column; gap: .15rem; }
.stat-k { font-size: .68rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: var(--muted); }
.stat-v { font-size: 1.05rem; font-weight: 700; }
.card .stat, .chart-card .stat { background: var(--surface-2); }

/* Weather hero card */
.wx { border-radius: 28px; padding: 1.4rem 1.4rem 1.2rem 1.4rem; border: 1px solid var(--line); box-shadow: var(--shadow); margin-bottom: .9rem; animation: rise .45s ease both; }
.wx-row { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
.wx-place { font-family: var(--serif); font-size: clamp(1.8rem, 6vw, 2.4rem); line-height: 1.05; margin-top: .3rem; }
.wx-place span { font-family: var(--sans); font-size: .8rem; font-weight: 700; color: var(--muted); margin-left: .5rem; letter-spacing: .08em; }
.wx-desc { color: var(--ink-2); font-weight: 500; text-transform: capitalize; margin-top: .2rem; }
.wx-badge { flex: none; width: 3.6rem; height: 3.6rem; border-radius: 50%; display: grid; place-items: center; font-size: 1.9rem; background: rgba(255,255,255,.78); box-shadow: 0 6px 18px -8px rgba(11,31,51,.3); }
.wx-temp { font-family: var(--serif); font-size: clamp(5rem, 20vw, 8rem); line-height: .9; letter-spacing: -0.03em; margin: 1rem 0 1.2rem 0; }
.wx-temp sup { font-family: var(--sans); font-size: .22em; font-weight: 700; vertical-align: top; position: relative; top: .55em; margin-left: .1em; color: var(--ink-2); letter-spacing: 0; }

/* Forecast */
.fc-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(92px, 1fr)); gap: .5rem; }
.fc-day { background: var(--surface-2); border-radius: 18px; padding: .8rem .4rem; text-align: center; transition: transform .2s; }
.fc-day:hover { transform: translateY(-3px); }
.fc-day.today { background: var(--accent-soft); box-shadow: inset 0 0 0 1.5px rgba(37,99,235,.35); }
.fc-name { font-size: .72rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; color: var(--muted); }
.fc-emoji { font-size: 1.7rem; margin: .35rem 0; }
.fc-hi { font-weight: 800; font-size: 1.1rem; }
.fc-lo { color: var(--muted); font-weight: 600; font-size: .9rem; }
.fc-rain { color: var(--accent); font-weight: 600; font-size: .75rem; margin-top: .35rem; }

/* Air quality */
.aq-pill { color: #fff; font-weight: 700; font-size: .8rem; border-radius: 999px; padding: .35rem .9rem; white-space: nowrap; }
.aq-scale { display: grid; grid-template-columns: repeat(5, 1fr); gap: 5px; }
.aq-scale span { height: 10px; border-radius: 99px; }
.aq-sub { color: var(--muted); font-size: .85rem; margin-top: .7rem; }

/* Local guide */
.season { background: #FFF4D6; color: #8A5A00; font-weight: 700; font-size: .78rem; border-radius: 999px; padding: .35rem .85rem; }
.lead { font-size: 1.02rem; line-height: 1.75; color: var(--ink-2); margin: 0; }
.pl-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: .9rem; }
.pl { background: var(--solid); border-radius: 20px; overflow: hidden; border: 1px solid var(--line); transition: transform .2s, box-shadow .2s; }
.pl:hover { transform: translateY(-3px); box-shadow: var(--shadow); }
.pl-pic { position: relative; aspect-ratio: 4 / 3; background: linear-gradient(135deg, #D8EBFF, #FFE8D2); overflow: hidden; }
.pl-pic img { width: 100%; height: 100%; object-fit: cover; display: block; transition: transform .4s; }
.pl:hover .pl-pic img { transform: scale(1.05); }
.pl-ph { display: grid; place-items: center; height: 100%; font-size: 2rem; }
.pl-num { position: absolute; top: .6rem; left: .6rem; background: rgba(255,255,255,.92); border-radius: 999px; padding: .15rem .6rem; font-size: .72rem; font-weight: 800; letter-spacing: .08em; }
.pl-body { padding: .85rem .95rem 1rem .95rem; }
.pl-name { font-weight: 700; }
.pl-why { color: var(--muted); font-size: .88rem; line-height: 1.55; margin-top: .3rem; }

[data-testid="stAlert"] { border-radius: 14px; }

@media (max-width: 640px) {
    .block-container { padding: 1rem .8rem 8rem .8rem; }
    .hero { padding-top: 1.8rem; }
    .hero::before { width: 96px; height: 96px; }
    .card, .wx, .chart-card { padding: 1.05rem; border-radius: 22px; }
    .fc-grid { grid-auto-flow: column; grid-auto-columns: 84px; grid-template-columns: none; overflow-x: auto; padding-bottom: .3rem; }
    .pl-grid { grid-auto-flow: column; grid-auto-columns: 82%; grid-template-columns: none; overflow-x: auto; scroll-snap-type: x mandatory; padding-bottom: .4rem; }
    .pl { scroll-snap-align: start; }
}
"""
st.markdown("<style>" + BASE_CSS + bg_css(st.session_state.bg) + "</style>", unsafe_allow_html=True)

st.markdown(
    '<div class="topbar"><div class="logo"><div class="logo-dot"></div>'
    f'<div class="logo-word">{APP_NAME}</div></div>'
    f'<div class="engine">{html.escape(CFG["label"])} · {html.escape(MODEL)}</div></div>',
    unsafe_allow_html=True,
)

_missing = [n for n, v in [(CFG["key_env"], LLM_KEY), ("WEATHER_API_KEY", WEATHER_API_KEY)] if not v]
if _missing:
    st.error(
        f"Missing in your .env file: {', '.join(_missing)}. "
        f"Provider in use: {PROVIDER} (set LLM_PROVIDER=groq, gemini or claude). "
        "Save the file next to app.py and restart Streamlit."
    )
    st.stop()


# ------------------------------------------------------------------
# Card builders (pure functions returning HTML)
# ------------------------------------------------------------------
def clean(s) -> str:
    """Escape text and collapse whitespace so it is safe inside our HTML."""
    return html.escape(" ".join(str(s).split()))


def strip(items: list) -> str:
    cells = "".join(
        f'<div class="stat"><span class="stat-k">{clean(k)}</span><span class="stat-v">{clean(v)}</span></div>'
        for k, v in items
    )
    return f'<div class="strip">{cells}</div>'


def pick_icon(main: str, icon_code: str) -> str:
    if main == "Clear":
        return "🌙" if str(icon_code).endswith("n") else "☀️"
    return {
        "Clouds": "☁️", "Rain": "🌧️", "Drizzle": "🌦️", "Thunderstorm": "⛈️",
        "Snow": "❄️", "Mist": "🌫️", "Fog": "🌫️", "Haze": "🌫️", "Smoke": "🌫️",
        "Dust": "🌪️", "Sand": "🌪️", "Tornado": "🌪️",
    }.get(main, "🌡️")


def temp_background(t_c: float) -> str:
    if t_c <= 10:
        return "linear-gradient(145deg, #E1ECFF, #F4F8FF)"
    if t_c <= 20:
        return "linear-gradient(145deg, #DDF2F8, #F0F7FF)"
    if t_c <= 27:
        return "linear-gradient(145deg, #E0F5EB, #F0F9FD)"
    if t_c <= 33:
        return "linear-gradient(145deg, #FFF0D3, #FFF9EC)"
    return "linear-gradient(145deg, #FFE3D0, #FFF2EA)"


def weather_card(d: dict, units: str) -> str:
    w, m = d["weather"][0], d["main"]
    imperial = units == "imperial"
    sym = "F" if imperial else "C"
    t_c = (m["temp"] - 32) * 5 / 9 if imperial else m["temp"]
    wind_raw = d.get("wind", {}).get("speed", 0)
    wind = f"{round(wind_raw)} mph" if imperial else f"{round(wind_raw * 3.6)} km/h"
    vis_m = d.get("visibility", 0)
    vis = f"{round(vis_m / 1609, 1)} mi" if imperial else f"{round(vis_m / 1000, 1)} km"
    stats = [
        ("Feels like", f"{round(m.get('feels_like', m['temp']))}°{sym}"),
        ("Humidity", f"{m.get('humidity', '-')}%"),
        ("Wind", wind),
        ("Pressure", f"{m.get('pressure', '-')} hPa"),
        ("Visibility", vis),
    ]
    country = clean(d.get("sys", {}).get("country", ""))
    country_html = f"<span>{country}</span>" if country else ""
    return (
        f'<div class="wx" style="background:{temp_background(t_c)}">'
        f'<div class="wx-row"><div><div class="eyebrow">Right now</div>'
        f'<div class="wx-place">{clean(d.get("name", ""))}{country_html}</div>'
        f'<div class="wx-desc">{clean(w.get("description", ""))}</div></div>'
        f'<div class="wx-badge">{pick_icon(w.get("main", ""), w.get("icon", ""))}</div></div>'
        f'<div class="wx-temp">{round(m["temp"])}<sup>°{sym}</sup></div>'
        f'{strip(stats)}</div>'
    )


def chart_head(place: str, now: float, feels: float, hi: float, lo: float, sym: str) -> str:
    stats = [
        ("Now", f"{round(now)}°{sym}"),
        ("Feels like", f"{round(feels)}°{sym}"),
        ("Highest", f"{round(hi)}°{sym}"),
        ("Lowest", f"{round(lo)}°{sym}"),
    ]
    return (
        f'<div class="chart-card"><div class="eyebrow">Temperature · next 5 days</div>'
        f'<div class="card-title" style="margin-bottom:1rem">{clean(place)}</div>{strip(stats)}'
        f'<div class="chart-note" style="margin-top:1rem">'
        f'<span><i class="dot" style="background:#2563EB"></i>Temperature</span>'
        f'<span><i class="dot" style="background:#F59E0B"></i>Daily high</span>'
        f'<span><i class="dot" style="background:#0EA5E9"></i>Daily low</span></div></div>'
    )


def forecast_card(f: dict) -> str:
    today = datetime.now().strftime("%Y-%m-%d")
    tiles = ""
    for day in sorted(f["daily"]):
        d = f["daily"][day]
        name = "Today" if day == today else datetime.strptime(day, "%Y-%m-%d").strftime("%a")
        cls = "fc-day today" if day == today else "fc-day"
        tiles += (
            f'<div class="{cls}"><div class="fc-name">{name}</div>'
            f'<div class="fc-emoji">{pick_icon(d["main"], d["icon"])}</div>'
            f'<div class="fc-hi">{d["max"]}°</div><div class="fc-lo">{d["min"]}°</div>'
            f'<div class="fc-rain">💧 {d["rain"]}%</div></div>'
        )
    return (
        f'<div class="card"><div class="card-head"><div><div class="eyebrow">5-day forecast</div>'
        f'<div class="card-title">{clean(f["city"])}</div></div></div>'
        f'<div class="fc-grid">{tiles}</div></div>'
    )


AQI_LEVELS = {
    1: ("Good", "#22B573"),
    2: ("Fair", "#8BC34A"),
    3: ("Moderate", "#F5B301"),
    4: ("Poor", "#F57C00"),
    5: ("Very poor", "#E53935"),
}


def aqi_card(r: dict) -> str:
    label, color = AQI_LEVELS.get(r["aqi"], ("Unknown", "#90A4AE"))
    segs = "".join(
        f'<span style="background:{c};opacity:{1 if i == r["aqi"] else 0.22}"></span>'
        for i, (_, c) in AQI_LEVELS.items()
    )
    return (
        f'<div class="card"><div class="card-head" style="margin-bottom:.9rem"><div>'
        f'<div class="eyebrow">Air quality</div><div class="card-title">{clean(r["city"])}</div></div>'
        f'<span class="aq-pill" style="background:{color}">{label}</span></div>'
        f'<div class="aq-scale">{segs}</div>'
        f'<div class="aq-sub">PM2.5 {r["pm2_5"]} µg/m³ · PM10 {r["pm10"]} µg/m³</div></div>'
    )


def guide_card(city: str, season: str, speciality: str, places: list, images: list) -> str:
    items = ""
    for n, ((name, why), img) in enumerate(zip(places, images), start=1):
        if img:
            pic = f'<img src="{html.escape(img, quote=True)}" alt="{clean(name)}" loading="lazy">'
        else:
            pic = '<div class="pl-ph">📍</div>'
        items += (
            f'<div class="pl"><div class="pl-pic">{pic}<span class="pl-num">0{n}</span></div>'
            f'<div class="pl-body"><div class="pl-name">{clean(name)}</div>'
            f'<div class="pl-why">{clean(why)}</div></div></div>'
        )
    return (
        f'<div class="card"><div class="card-head"><div><div class="eyebrow">Local guide</div>'
        f'<div class="card-title">{clean(city)}</div></div>'
        f'<span class="season">{clean(season)}</span></div>'
        f'<p class="lead">{clean(speciality)}</p>'
        f'<div class="eyebrow" style="margin:1.4rem 0 .7rem 0">Worth visiting right now</div>'
        f'<div class="pl-grid">{items}</div></div>'
    )


def temp_chart(c: dict):
    """Modern temperature chart: smooth gradient area, glowing line, daily high/low markers, tooltip."""
    df = pd.DataFrame({"time": pd.to_datetime(c["times"]), "temp": c["temps"]})
    df["day"] = df["time"].dt.strftime("%Y-%m-%d")
    unit = c["label"].split("(")[-1].rstrip(")")  # e.g. °C

    ymin, ymax = df["temp"].min(), df["temp"].max()
    pad = max((ymax - ymin) * 0.25, 1.5)
    y = alt.Y(
        "temp:Q", title=None,
        scale=alt.Scale(domain=[ymin - pad, ymax + pad], nice=True),
        axis=alt.Axis(format=".0f", labelColor="#5B7083", labelFontSize=11, tickCount=4,
                      grid=True, gridColor="rgba(11,31,51,.07)", gridDash=[3, 4],
                      domain=False, ticks=False, labelPadding=8),
    )
    x = alt.X(
        "time:T", title=None,
        axis=alt.Axis(format="%a", labelColor="#5B7083", labelFontSize=11, tickCount="day",
                      grid=False, domain=False, ticks=False, labelPadding=8),
    )

    area = alt.Chart(df).mark_area(
        interpolate="monotone",
        color=alt.Gradient(
            gradient="linear",
            stops=[alt.GradientStop(color="rgba(37,99,235,0.38)", offset=0),
                   alt.GradientStop(color="rgba(124,58,237,0.02)", offset=1)],
            x1=1, x2=1, y1=0, y2=1,
        ),
    ).encode(x=x, y=y)

    line = alt.Chart(df).mark_line(
        interpolate="monotone", strokeWidth=3, strokeCap="round",
        color=alt.Gradient(
            gradient="linear",
            stops=[alt.GradientStop(color="#2563EB", offset=0), alt.GradientStop(color="#7C3AED", offset=1)],
            x1=0, x2=1, y1=0, y2=0,
        ),
    ).encode(x=x, y=y)

    # Hover: nearest point + vertical rule + tooltip
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["time"], empty=False, clear="pointerout")
    tip = [
        alt.Tooltip("time:T", title="Time", format="%a %d %b, %H:%M"),
        alt.Tooltip("temp:Q", title=f"Temp ({unit})", format=".1f"),
    ]
    selectors = alt.Chart(df).mark_point(opacity=0).encode(x=x, tooltip=tip).add_params(nearest)
    rule = alt.Chart(df).mark_rule(color="#2563EB", strokeDash=[4, 4], opacity=.5).encode(
        x=x, opacity=alt.condition(nearest, alt.value(.55), alt.value(0)))
    dots = alt.Chart(df).mark_point(filled=True, size=90, color="#2563EB", stroke="white", strokeWidth=2.5).encode(
        x=x, y=y, opacity=alt.condition(nearest, alt.value(1), alt.value(0)))

    # Highest and lowest markers
    hi_row = df.loc[[df["temp"].idxmax()]].assign(kind="high")
    lo_row = df.loc[[df["temp"].idxmin()]].assign(kind="low")
    ext = pd.concat([hi_row, lo_row])
    ext_pts = alt.Chart(ext).mark_point(filled=True, size=110, stroke="white", strokeWidth=2.5).encode(
        x=x, y=y,
        color=alt.Color("kind:N", legend=None,
                        scale=alt.Scale(domain=["high", "low"], range=["#F59E0B", "#0EA5E9"])),
    )
    ext_lbl = alt.Chart(ext).mark_text(dy=-16, fontSize=12, fontWeight=700, color="#0B1F33").encode(
        x=x, y=y, text=alt.Text("temp:Q", format=".0f"))

    return (
        alt.layer(area, line, selectors, rule, dots, ext_pts, ext_lbl)
        .properties(width="container", height=270, background="transparent", padding={"left": 4, "right": 12, "top": 14, "bottom": 4})
        .configure_view(stroke=None)
        .configure(font="Plus Jakarta Sans")
    )


def render_cards(cards: list):
    for c in cards:
        if c["kind"] == "html":
            st.markdown(c["html"], unsafe_allow_html=True)
        else:  # temperature chart: summary card, then the graph
            st.markdown(c["head"], unsafe_allow_html=True)
            st.altair_chart(temp_chart(c))


# ------------------------------------------------------------------
# Data sources: OpenWeatherMap (cached 10 min) and Wikipedia pictures
# ------------------------------------------------------------------
@st.cache_resource
def get_cache() -> dict:
    return {}


CACHE = get_cache()


def api(url: str, ttl: int = 600, **params):
    key = (url, tuple(sorted(params.items())))
    hit = CACHE.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    r = requests.get(url, params={**params, "appid": WEATHER_API_KEY}, timeout=10)
    data = r.json()
    if r.ok:
        CACHE[key] = (time.time(), data)
    return data


def owm(path: str, **params) -> dict:
    return api(f"https://api.openweathermap.org/data/2.5/{path}", **params)


def locate(city: str) -> dict:
    """Resolve a city, state or region (e.g. Mizoram) to coordinates."""
    try:
        g = api("https://api.openweathermap.org/geo/1.0/direct", q=city, limit=1)
    except requests.RequestException:
        g = None
    if isinstance(g, list) and g:
        p = g[0]
        return {
            "params": {"lat": p["lat"], "lon": p["lon"]},
            "name": p.get("name", city),
            "country": p.get("country", ""),
        }
    return {"params": {"q": city}, "name": city, "country": ""}


def named(data: dict, loc: dict) -> dict:
    """Copy of a weather response labelled with the place the user asked for."""
    data = dict(data)
    data["name"] = loc["name"]
    data["sys"] = dict(data.get("sys", {}))
    if loc["country"]:
        data["sys"]["country"] = loc["country"]
    return data


def wiki_image(query: str):
    """Thumbnail of the best-matching Wikipedia article (no API key needed)."""
    key = ("wiki", query)
    hit = CACHE.get(key)
    if hit and time.time() - hit[0] < 6 * 3600:
        return hit[1]
    img = None
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "generator": "search", "gsrsearch": query, "gsrlimit": 1,
                "prop": "pageimages", "piprop": "thumbnail", "pithumbsize": 600,
                "format": "json", "origin": "*",
            },
            headers={"User-Agent": "BreezeWeatherApp/1.0 (learning project)"},
            timeout=8,
        )
        for page in r.json().get("query", {}).get("pages", {}).values():
            img = page.get("thumbnail", {}).get("source")
    except Exception:
        img = None
    CACHE[key] = (time.time(), img)
    return img


# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------
TOOL_LABELS = {
    "get_weather": "Checking the weather",
    "show_temperature_chart": "Drawing the temperature graph",
    "get_forecast": "Loading the forecast",
    "get_air_quality": "Checking air quality",
    "show_place_guide": "Finding places to visit",
}


def build_agent(captured: dict, units: str) -> Agent:
    # New client per request: asyncio.run() creates a fresh event loop each time.
    client = AsyncOpenAI(base_url=CFG["base_url"], api_key=LLM_KEY)
    model = OpenAIChatCompletionsModel(model=MODEL, openai_client=client)
    sym = "F" if units == "imperial" else "C"

    @function_tool
    def get_weather(city: str) -> str:
        """Current overall weather (conditions, temperature, humidity, wind) for a city, state or region."""
        loc = locate(city)
        try:
            data = owm("weather", **loc["params"], units=units)
        except requests.RequestException as e:
            return json.dumps({"error": str(e)})
        if str(data.get("cod")) != "200":
            return json.dumps(data)
        data = named(data, loc)
        w = data["weather"][0]
        captured["cards"].append({"kind": "html", "html": weather_card(data, units)})
        captured["bg"] = bg_key(w.get("main", ""), w.get("icon", ""))
        return json.dumps(data)

    @function_tool
    def show_temperature_chart(city: str) -> str:
        """Temperature graph for the next 5 days plus the current temperature. Use when the user asks about temperature."""
        loc = locate(city)
        try:
            now = owm("weather", **loc["params"], units=units)
            fc = owm("forecast", **loc["params"], units=units)
        except requests.RequestException as e:
            return json.dumps({"error": str(e)})
        if str(now.get("cod")) != "200":
            return json.dumps(now)
        if str(fc.get("cod")) != "200":
            return json.dumps(fc)
        times = [it["dt_txt"] for it in fc["list"]]
        temps = [round(it["main"]["temp"], 1) for it in fc["list"]]
        hi, lo = max(temps), min(temps)
        head = chart_head(loc["name"], now["main"]["temp"], now["main"].get("feels_like", now["main"]["temp"]), hi, lo, sym)
        captured["cards"].append(
            {"kind": "chart", "head": head, "label": f"Temperature (°{sym})", "times": times, "temps": temps}
        )
        return json.dumps({
            "place": loc["name"],
            "now": round(now["main"]["temp"]),
            "feels_like": round(now["main"].get("feels_like", now["main"]["temp"])),
            "humidity": now["main"].get("humidity"),
            "condition": now["weather"][0]["description"],
            "highest_next_5_days": round(hi),
            "lowest_next_5_days": round(lo),
            "unit": sym,
        })

    @function_tool
    def get_forecast(city: str) -> str:
        """5-day forecast tiles (daily high, low and rain chance) for a city, state or region."""
        loc = locate(city)
        try:
            data = owm("forecast", **loc["params"], units=units)
        except requests.RequestException as e:
            return json.dumps({"error": str(e)})
        if str(data.get("cod")) != "200":
            return json.dumps(data)
        daily: dict = {}
        for it in data["list"]:
            day, hour = it["dt_txt"][:10], it["dt_txt"][11:13]
            w = it["weather"][0]
            d = daily.setdefault(
                day,
                {"min": 1e9, "max": -1e9, "rain": 0,
                 "main": w["main"], "icon": w["icon"], "desc": w["description"]},
            )
            d["min"] = min(d["min"], it["main"]["temp_min"])
            d["max"] = max(d["max"], it["main"]["temp_max"])
            d["rain"] = max(d["rain"], round(it.get("pop", 0) * 100))
            if hour == "12":  # midday conditions represent the day
                d.update(main=w["main"], icon=w["icon"], desc=w["description"])
        for d in daily.values():
            d["min"], d["max"] = round(d["min"]), round(d["max"])
        result = {"city": loc["name"], "daily": daily}
        captured["cards"].append({"kind": "html", "html": forecast_card(result)})
        return json.dumps(result)

    @function_tool
    def get_air_quality(city: str) -> str:
        """Air quality index (1 good to 5 very poor) and PM2.5/PM10 for a city, state or region."""
        loc = locate(city)
        try:
            geo = owm("weather", **loc["params"], units=units)
            if str(geo.get("cod")) != "200":
                return json.dumps(geo)
            aq = owm("air_pollution", lat=geo["coord"]["lat"], lon=geo["coord"]["lon"])
        except requests.RequestException as e:
            return json.dumps({"error": str(e)})
        item = aq["list"][0]
        result = {
            "city": loc["name"],
            "aqi": item["main"]["aqi"],
            "pm2_5": item["components"]["pm2_5"],
            "pm10": item["components"]["pm10"],
        }
        captured["cards"].append({"kind": "html", "html": aqi_card(result)})
        return json.dumps(result)

    @function_tool
    def show_place_guide(
        city: str,
        season_label: str,
        speciality: str,
        place1_name: str,
        place1_why: str,
        place2_name: str,
        place2_why: str,
        place3_name: str,
        place3_why: str,
    ) -> str:
        """Show a visitor guide card with the place's speciality and 3 places to visit, with pictures.
        Call this only AFTER get_weather or show_temperature_chart has returned a result.

        Args:
            city: The place the user asked about, as the user wrote it.
            season_label: Short label for the current season and weather, e.g. "Cool, misty autumn".
            speciality: 2 to 3 sentences on what the place is famous for (culture, food, nature), linked to the current temperature.
            place1_name: Name of a real, well-known place to visit in or near it that suits this weather.
            place1_why: Exactly two short sentences on why to visit it in this weather.
            place2_name: Name of a second real, well-known place that suits this weather.
            place2_why: Exactly two short sentences on why to visit it in this weather.
            place3_name: Name of a third real, well-known place that suits this weather.
            place3_why: Exactly two short sentences on why to visit it in this weather.
        """
        places = [(place1_name, place1_why), (place2_name, place2_why), (place3_name, place3_why)]
        with ThreadPoolExecutor(max_workers=3) as ex:
            images = list(ex.map(lambda p: wiki_image(f"{p[0]} {city}"), places))
        captured["cards"].append(
            {"kind": "html", "html": guide_card(city, season_label, speciality, places, images)}
        )
        return "The guide card is now shown to the user. Do not list the places again."

    unit_name = "Fahrenheit" if units == "imperial" else "Celsius"
    today = datetime.now().strftime("%d %B %Y")
    return Agent(
        name="Breeze Weather Agent",
        instructions=(
            f"You are Breeze, a friendly weather and travel assistant. Today is {today}. "
            "The place can be a city, a state or a region (for example Mizoram). Choose tools like this: "
            "(1) If the user asks about TEMPERATURE (how hot or cold, degrees), call show_temperature_chart. "
            "(2) If the user asks about the WEATHER in general (weather, raining, sunny, how is it), call get_weather. "
            "(3) For a single place, once you have received the result of (1) or (2), call show_place_guide exactly once. "
            "Never call show_place_guide in the same step as the weather tool, because you need the real temperature first. "
            "(4) For upcoming days or trips call get_forecast. For air quality or pollution call get_air_quality. "
            "(5) For comparisons call get_weather once per place and skip show_place_guide. "
            "For show_place_guide, write the speciality so it connects to the current temperature and season, and pick three "
            "real, well-known places in or near the place that suit that temperature and season. Each 'why' must be exactly "
            "two short sentences. Never invent places. "
            "The cards already show all numbers, places and pictures, so your own reply must be 1 to 3 short, warm sentences "
            f"with the key takeaway and one practical tip. Do not list the places or repeat the numbers. Use {unit_name}. "
            "If a place is not found, say so and ask for a different spelling. "
            "If the question is not about weather or travel, politely steer back to weather."
        ),
        tools=[get_weather, show_temperature_chart, get_forecast, get_air_quality, show_place_guide],
        model=model,
    )


def ask_agent(user_input: str, units: str, status, out):
    captured = {"cards": [], "bg": None}
    agent = build_agent(captured, units)
    history = st.session_state.agent_history + [{"role": "user", "content": user_input}]

    async def run():
        result = Runner.run_streamed(agent, history)
        text = ""
        async for ev in result.stream_events():
            if ev.type == "run_item_stream_event" and ev.name == "tool_called":
                tool = getattr(ev.item.raw_item, "name", "tool")
                status.update(label=f"{TOOL_LABELS.get(tool, 'Working')}...")
                text = ""
            elif ev.type == "raw_response_event" and isinstance(ev.data, ResponseTextDeltaEvent):
                text += ev.data.delta
                out.markdown(text + "▌")
        return result, text

    result, text = asyncio.run(run())
    reply = result.final_output or text
    out.markdown(reply)
    st.session_state.agent_history = result.to_input_list()
    return reply, captured["cards"], captured["bg"]


def chat_markdown() -> str:
    """Readable Markdown transcript for the Export button."""
    stamp = datetime.now().strftime("%d %b %Y, %H:%M")
    n = len(st.session_state.messages)
    lines = [f"# {APP_NAME} · Chat export", "", f"*Exported {stamp} · {n} messages*", "", "---", ""]
    for m in st.session_state.messages:
        who = "🧑 You" if m["role"] == "user" else f"🌤️ {APP_NAME}"
        lines += [f"### {who}", "", m["content"], ""]
    return "\n".join(lines)


USER_AVATAR, BOT_AVATAR = "🧑", "🌤️"

# ------------------------------------------------------------------
# Hero (empty chat only)
# ------------------------------------------------------------------
has_chat = bool(st.session_state.messages)

if not has_chat and not st.session_state.pending:
    st.markdown(
        '<div class="hero"><div class="eyebrow">Live weather · forecasts · local guides</div>'
        '<div class="hero-title">The sky, <em>explained.</em></div>'
        '<p class="hero-sub">Ask about any city or state. Get live conditions, a temperature graph '
        "and three places worth visiting this season.</p></div>",
        unsafe_allow_html=True,
    )

# ------------------------------------------------------------------
# Action bar: units · New chat · Export chat (always visible)
# ------------------------------------------------------------------
with st.container(key="actionbar"):
    c1, c2, c3 = st.columns([1.4, 1, 1])
    if hasattr(st, "segmented_control"):
        units_label = c1.segmented_control(
            "Units", ["°C", "°F"], default="°C", label_visibility="collapsed", key="units_ctl"
        ) or "°C"
    else:
        units_label = c1.radio("Units", ["°C", "°F"], horizontal=True, label_visibility="collapsed")
    if c2.button("✨ New chat", key="new_chat", disabled=not has_chat):
        st.session_state.messages = []
        st.session_state.agent_history = []
        st.session_state.pending = None
        st.session_state.bg = "default"
        st.rerun()
    c3.download_button(
        "⬇ Export chat",
        chat_markdown() if has_chat else " ",
        file_name=f"breeze_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        key="export_chat",
        disabled=not has_chat,
    )
units = "imperial" if units_label == "°F" else "metric"

# Suggestions on an empty chat
if not has_chat and not st.session_state.pending:
    st.markdown('<div class="eyebrow try">Try asking</div>', unsafe_allow_html=True)
    chips = {
        "☀️  Weather in Mizoram": "What is the weather in Mizoram?",
        "🌡️  Temperature in Pune": "What is the temperature in Pune?",
        "📅  Pune 5-day forecast": "Give me the 5-day forecast for Pune",
        "🍃  Delhi air quality": "How is the air quality in Delhi?",
    }
    items = list(chips.items())
    for row in (items[:2], items[2:]):
        for col, (label, q) in zip(st.columns(2), row):
            if col.button(label, key=f"chip_{label}"):
                st.session_state.pending = q
                st.rerun()

# ------------------------------------------------------------------
# History
# ------------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=USER_AVATAR if msg["role"] == "user" else BOT_AVATAR):
        render_cards(msg.get("cards", []))
        st.markdown(msg["content"])

# ------------------------------------------------------------------
# New input
# ------------------------------------------------------------------
prompt = st.chat_input("Ask about weather, temperature, forecasts or air quality...")
if st.session_state.pending:
    prompt = st.session_state.pending
    st.session_state.pending = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    new_bg = None
    with st.chat_message("assistant", avatar=BOT_AVATAR):
        slot = st.container()
        status = st.status("Thinking...", expanded=False)
        out = st.empty()
        cards = []
        try:
            reply, cards, new_bg = ask_agent(prompt, units, status, out)
            status.update(label="Done", state="complete")
        except Exception as e:
            err = str(e)
            reply = f"Something went wrong: `{err}`"
            if "404" in err or "not found" in err.lower():
                reply += f"\n\nThe model `{MODEL}` isn't available for your key. Set `{CFG['model_env']}` in .env to another model."
            elif "401" in err or "403" in err or "api key" in err.lower():
                reply += f"\n\nCheck that `{CFG['key_env']}` in .env is correct."
            out.markdown(reply)
            status.update(label="Failed", state="error")
        with slot:
            render_cards(cards)

    st.session_state.messages.append({"role": "assistant", "content": reply, "cards": cards})

    # Overall weather asked -> switch the page background to match, then redraw
    if new_bg and new_bg != st.session_state.bg:
        st.session_state.bg = new_bg
    # Rerun so the action bar (New chat / Export) updates with the new message
    st.rerun()