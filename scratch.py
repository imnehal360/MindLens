import sys

with open("frontend/app.py", "r") as f:
    content = f.read()

# Add pandas
content = content.replace("from datetime import datetime", "from datetime import datetime\nimport pandas as pd")

# Update sidebar
old_nav = '["🔮  Fusion Analysis", "💬  Text Only", "🎙️  Audio Only",\n             "📸  Face Only", "📊  Report & Trend"]'
new_nav = '["🔮  Fusion Analysis", "📊  Dashboard"]'
content = content.replace(old_nav, new_nav)

# Find the start of PAGE: TEXT ONLY
cut_idx = content.find("# ─────────────────────────────────────────\n# PAGE: TEXT ONLY")

if cut_idx == -1:
    print("Could not find PAGE: TEXT ONLY string!")
    sys.exit(1)

new_dashboard = """# ─────────────────────────────────────────
# PAGE: DASHBOARD
# ─────────────────────────────────────────
def page_dashboard():
    st.markdown('<div class="hero-title" style="font-size:1.6rem;font-family:var(--font-head);font-weight:800;background:linear-gradient(90deg,#6ee7b7,#818cf8);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:1.2rem;">📊 Mental Health Dashboard</div>', unsafe_allow_html=True)

    col_rep, col_trend = st.columns(2)

    with col_rep:
        if st.button("Load Dashboard Data", use_container_width=True):
            with st.spinner("Fetching data…"):
                r = api_report()
            st.session_state["_report"] = r

    with col_trend:
        if st.button("Load Trend Forecast", use_container_width=True):
            with st.spinner("Fetching trend…"):
                t = api_trend()
            st.session_state["_trend"] = t

    # Trend card
    if "_trend" in st.session_state and st.session_state["_trend"]:
        t = st.session_state["_trend"]
        if "detail" in t:
            st.error(t["detail"])
        else:
            future = t.get("future_risk")
            pct = round(future * 100, 1) if future is not None else None
            st.markdown('<div style="height:.8rem"></div>', unsafe_allow_html=True)
            render_risk_gauge(pct, "Predicted Future Risk (LSTM)")

    # Report & Graphs
    if "_report" in st.session_state and st.session_state["_report"]:
        r = st.session_state["_report"]
        if "detail" in r:
            st.error(r["detail"])
        else:
            future = r.get("future_risk")
            status = r.get("status", "—")
            history = r.get("history", [])
            records = r.get("records_found", 0)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Sessions Found", records)
            col_b.metric("Status", status)
            col_c.metric("Future Risk", f"{round(future*100,1):.1f}" if future else "N/A")

            if history:
                st.markdown("---")
                st.markdown('<div class="section-label">Historical Trends</div>', unsafe_allow_html=True)
                
                # Parse history for Pandas
                data = []
                for h in reversed(history):
                    dt = datetime.fromisoformat(str(h.get("timestamp")).replace("Z", ""))
                    data.append({
                        "Time": dt.strftime("%b %d %H:%M"),
                        "Risk": round(h.get("risk", 0) * 100, 1),
                        "Depression": round(h.get("depression", 0) * 100, 1),
                        "Anxiety": round(h.get("anxiety", 0) * 100, 1),
                        "Stress": round(h.get("stress", 0) * 100, 1),
                    })
                
                df = pd.DataFrame(data).set_index("Time")
                
                st.markdown("**Overall Risk Over Time**")
                st.line_chart(df[["Risk"]], use_container_width=True)

                st.markdown("**Detailed Metrics Over Time**")
                st.line_chart(df[["Depression", "Anxiety", "Stress"]], use_container_width=True)

                st.markdown("---")
                st.markdown('<div class="section-label">Session History (last 7)</div>',
                            unsafe_allow_html=True)
                for h in reversed(history):
                    ts     = fmt_ts(h.get("timestamp", ""))
                    risk   = round(h.get("risk", 0) * 100, 1)
                    dep    = round(h.get("depression", 0) * 100, 1)
                    anx    = round(h.get("anxiety", 0) * 100, 1)
                    stress = round(h.get("stress", 0) * 100, 1)
                    pill   = risk_pill(risk)
                    st.markdown(f'''
                    <div class="hist-row">
                        <span class="hist-ts">{ts}</span>
                        <span>Depression <b>{dep}</b></span>
                        <span>Anxiety <b>{anx}</b></span>
                        <span>Stress <b>{stress}</b></span>
                        <span>Risk <b>{risk}</b> {pill}</span>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.info("No history yet — run a fusion analysis to start recording sessions.")

# ─────────────────────────────────────────
# MAIN ROUTER
# ─────────────────────────────────────────
def main():
    if not st.session_state.token:
        page_auth()
        return

    page = sidebar()

    if "Fusion" in page:
        page_fusion()
    elif "Dashboard" in page:
        page_dashboard()

if __name__ == "__main__":
    main()
"""

content = content[:cut_idx] + new_dashboard

with open("frontend/app.py", "w") as f:
    f.write(content)

print("Done")
