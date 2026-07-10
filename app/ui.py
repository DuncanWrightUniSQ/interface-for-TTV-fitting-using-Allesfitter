"""Shared Streamlit layout helpers."""

from __future__ import annotations

from collections.abc import Iterable

import streamlit as st


def page_header(title: str, body: str, actions: Iterable[str] = ()) -> None:
    left, right = st.columns([0.68, 0.32], vertical_alignment="top")
    with left:
        st.header(title)
        st.caption(body)
    with right:
        st.write("")
        for action in actions:
            st.button(action, use_container_width=True, disabled=True)


def pending_panel(title: str, lines: Iterable[str]) -> None:
    st.subheader(title)
    for line in lines:
        st.checkbox(line, value=False, disabled=True)


def metric_strip(items: Iterable[tuple[str, str, str]]) -> None:
    item_list = list(items)
    columns = st.columns(len(item_list))
    for column, (label, value, delta) in zip(columns, item_list):
        column.metric(label, value, delta=delta or None)
