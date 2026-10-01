"""Interface de l'app : composants HTML a styles INLINE (aucune feuille de
style globale). Choix delibere : une regle CSS globale injectee via une
balise <style> s'est averee ne pas se monter de facon fiable au premier
chargement dans cet environnement (bug non resolu malgre investigation
approfondie, visible y compris avec st.markdown et st.html). Le style inline
sur chaque element est verifie fonctionner de facon fiable, a chaque essai.

Le theme de base (fond, couleur de texte, couleur primaire des boutons/
widgets natifs) est gere nativement par .streamlit/config.toml, qui ne
depend d'aucune injection HTML/CSS cote client.
"""
import html

import streamlit as st

NAVY = "#0F1E33"
NAVY_2 = "#1B3A5C"
GOLD = "#B8925A"
GREEN = "#1F6F4A"
WINE = "#8A3B3B"
LINE = "#E3E7EE"
TEXT = "#1F2937"
TEXT_SOFT = "#5B6779"
FONT = "Inter, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif"


def esc(text) -> str:
    return html.escape(str(text), quote=False)


def inject_css() -> None:
    """Conservee pour compatibilite des imports existants ; ne fait plus
    rien (voir le docstring du module pour la raison)."""
    return None


def hero(kicker: str, title: str, text: str) -> None:
    st.html(
        f'<div style="background:linear-gradient(135deg,{NAVY} 0%,{NAVY_2} 100%);'
        f'border-radius:16px;padding:36px 40px;margin-bottom:18px;font-family:{FONT};">'
        f'<div style="font-size:12px;font-weight:600;letter-spacing:.14em;color:#D8B77E;'
        f'text-transform:uppercase;">{esc(kicker)}</div>'
        f'<h1 style="font-size:32px;font-weight:800;line-height:1.2;letter-spacing:-.02em;'
        f'margin:10px 0 12px 0;color:#fff;">{esc(title)}</h1>'
        f'<p style="font-size:15.5px;line-height:1.6;color:#CBD5E4;max-width:680px;margin:0;">'
        f'{esc(text)}</p></div>'
    )


def section(title: str) -> None:
    st.html(
        f'<div style="font-size:20px;font-weight:700;color:{NAVY};margin:22px 0 10px 0;'
        f'display:flex;align-items:center;gap:12px;font-family:{FONT};">'
        f'<span style="width:5px;height:20px;background:{GOLD};border-radius:3px;'
        f'display:inline-block;"></span>{esc(title)}</div>'
    )


def stats(items) -> None:
    cards = "".join(
        f'<div style="flex:1 1 180px;background:#fff;border:1px solid {LINE};'
        f'border-radius:12px;padding:14px 16px;font-family:{FONT};">'
        f'<div style="font-size:22px;font-weight:700;color:{NAVY};">{esc(v)}</div>'
        f'<div style="font-size:12.5px;color:{TEXT_SOFT};margin-top:4px;line-height:1.4;">'
        f'{esc(l)}</div></div>'
        for v, l in items
    )
    st.html(f'<div style="display:flex;flex-wrap:wrap;gap:12px;margin:14px 0;">{cards}</div>')


def card(title: str, body_html: str) -> None:
    st.html(
        f'<div style="background:#fff;border:1px solid {LINE};border-radius:12px;'
        f'padding:18px 20px;margin-bottom:14px;font-family:{FONT};">'
        f'<h3 style="font-size:15.5px;font-weight:700;color:{NAVY};margin:0 0 6px 0;">'
        f'{esc(title)}</h3>'
        f'<div style="font-size:14px;line-height:1.6;color:{TEXT};">{body_html}</div></div>'
    )


def reading(text_html: str) -> None:
    st.html(
        f'<div style="background:#fff;border:1px solid {LINE};border-left:4px solid {NAVY_2};'
        f'border-radius:8px;padding:12px 16px;font-size:13.8px;line-height:1.6;color:{TEXT};'
        f'margin:6px 0 14px 0;font-family:{FONT};">{text_html}</div>'
    )


def steps(items) -> None:
    rows = "".join(
        f'<div style="display:flex;gap:14px;padding:12px 0;border-top:1px solid {LINE};">'
        f'<div style="width:26px;height:26px;border-radius:50%;background:{NAVY_2};color:#fff;'
        f'font-size:12.5px;font-weight:700;display:flex;align-items:center;justify-content:center;'
        f'flex:none;">{i + 1}</div>'
        f'<div style="font-size:14.5px;line-height:1.55;color:{TEXT};padding-top:2px;">{t}</div>'
        f'</div>'
        for i, t in enumerate(items)
    )
    st.html(
        f'<div style="font-family:{FONT};border-bottom:1px solid {LINE};">{rows}</div>'
    )


def badge(text: str, kind: str = "ok") -> str:
    colors = {"ok": (GREEN, "#E3F3EA"), "warn": ("#9C6B1F", "#FBEFE3"), "bad": (WINE, "#FBE9E9")}
    fg, bg = colors.get(kind, colors["ok"])
    return (
        f'<span style="display:inline-block;font-size:11.5px;font-weight:600;padding:3px 10px;'
        f'border-radius:999px;margin-right:6px;color:{fg};background:{bg};">{esc(text)}</span>'
    )


def verdict(proba: float, threshold: float = 0.5) -> None:
    is_risk = proba >= threshold
    fg, bg = (WINE, "#FBE9E9") if is_risk else (GREEN, "#E3F3EA")
    label = "Sinistre jugé à risque de fraude" if is_risk else "Sinistre jugé peu suspect"
    st.html(
        f'<div style="border-radius:14px;padding:26px 28px;text-align:center;margin:10px 0 18px 0;'
        f'background:{bg};font-family:{FONT};">'
        f'<div style="font-size:42px;font-weight:800;line-height:1;color:{fg};">'
        f'{proba * 100:.1f}\u00a0%</div>'
        f'<div style="font-size:14.5px;font-weight:600;margin-top:8px;color:{TEXT};">'
        f'{esc(label)} \u2014 probabilité de fraude estimée</div></div>'
    )
