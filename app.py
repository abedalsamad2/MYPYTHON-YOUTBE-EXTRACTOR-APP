# pyright: reportMissingImports=false, reportMissingModuleSource=false
import gc
import json
import re
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, urlparse

import google.generativeai as genai
import pandas as pd
import streamlit as st
import yt_dlp
from google.generativeai import GenerativeModel
from groq import Groq
from youtube_transcript_api import YouTubeTranscriptApi


DATA_FILE = Path("processed_data.json")
CSV_FILE = Path("processed_data.csv")
TMP_AUDIO_DIR = Path("tmp_audio")
ARABIC_FONT_FILE = Path("Amiri-Regular.ttf")
BATCH_SIZE = 10
RATE_LIMIT_SECONDS = 2


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_state() -> None:
    if st.session_state.get("initialized"):
        return
    payload = load_progress()
    st.session_state.all_urls = payload.get("all_urls", [])
    st.session_state.records = payload.get("records", {})
    st.session_state.last_processed_index = payload.get("last_processed_index", -1)
    st.session_state.chat_history = []
    st.session_state.initialized = True


def load_progress() -> Dict:
    if not DATA_FILE.exists():
        return {"all_urls": [], "records": {}, "last_processed_index": -1}

    try:
        raw_text = DATA_FILE.read_text(encoding="utf-8").strip()
        if not raw_text:
            return {"all_urls": [], "records": {}, "last_processed_index": -1}
        data = json.loads(raw_text)
    except Exception:
        return {"all_urls": [], "records": {}, "last_processed_index": -1}

    # التأكد من أن البيانات عبارة عن قاموس (Dict) وليس قائمة (List)
    if not isinstance(data, dict):
        return {"all_urls": [], "records": {}, "last_processed_index": -1}

    all_urls = data.get("all_urls", [])
    raw_records = data.get("records", [])
    records_map: Dict[str, Dict] = {}

    # معالجة Records سواء كانت قاموساً أو قائمة
    if isinstance(raw_records, dict):
        for rec in raw_records.values():
            if isinstance(rec, dict) and rec.get("url"):
                records_map[rec["url"]] = rec
    elif isinstance(raw_records, list):
        for rec in raw_records:
            if isinstance(rec, dict) and rec.get("url"):
                records_map[rec["url"]] = rec

    last_processed_index = data.get("last_processed_index", -1)
    
    # تصحيح المؤشر لضمان عدم خروجه عن نطاق القائمة
    if all_urls:
        last_processed_index = min(max(-1, last_processed_index), len(all_urls) - 1)
    else:
        last_processed_index = -1

    return {
        "all_urls": all_urls,
        "records": records_map,
        "last_processed_index": last_processed_index,
    }


def save_progress(all_urls: List[str], records: Dict[str, Dict], last_processed_index: int) -> None:
    payload = {
        "all_urls": all_urls,
        "records": list(records.values()),
        "last_processed_index": last_processed_index,
        "last_updated_utc": utc_now_iso(),
    }
    tmp_file = DATA_FILE.with_suffix(".tmp")
    tmp_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_file.replace(DATA_FILE)

    df = pd.DataFrame(list(records.values())) if records else pd.DataFrame(
        columns=["url", "status", "method_used", "timestamp_utc", "video_id", "title", "error"]
    )
    df.to_csv(CSV_FILE, index=False, encoding="utf-8")


def infer_last_processed_index(all_urls: List[str], records: Dict[str, Dict]) -> int:
    last_index = -1
    for idx, url in enumerate(all_urls):
        if url in records:
            last_index = idx
    return last_index


def extract_video_id(url: str) -> Optional[str]:
    candidate = (url or "").strip()
    if not candidate:
        return None
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", candidate):
        return candidate

    parsed = urlparse(candidate)
    host = parsed.netloc.lower().replace("www.", "")
    path = parsed.path.strip("/")

    if host == "youtu.be" and path:
        video_id = path.split("/")[0]
        return video_id if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id) else None

    if host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if path == "watch":
            video_id = parse_qs(parsed.query).get("v", [None])[0]
            return video_id if video_id and re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id) else None
        if path.startswith("shorts/") or path.startswith("embed/"):
            parts = path.split("/")
            video_id = parts[1] if len(parts) > 1 else ""
            return video_id if re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id) else None
    return None


def canonical_url_from_id(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def parse_bulk_urls(text: str) -> List[str]:
    seen = set()
    normalized = []
    for token in re.split(r"[\s,]+", (text or "").strip()):
        if not token:
            continue
        video_id = extract_video_id(token)
        if not video_id:
            continue
        url = canonical_url_from_id(video_id)
        if url not in seen:
            seen.add(url)
            normalized.append(url)
    return normalized


def fetch_transcript_stage1(video_id: str) -> str:
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id, languages=["ar", "en", "en-US"])
        text = " ".join(seg.get("text", "").strip() for seg in segments if seg.get("text"))
        if text.strip():
            return text.strip()
    except Exception:
        pass

    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id)
        text = " ".join(seg.get("text", "").strip() for seg in segments if seg.get("text"))
        if text.strip():
            return text.strip()
    except Exception as e:
        raise RuntimeError(f"Transcript API failed: {e}")
    
    raise RuntimeError("Empty transcript from youtube-transcript-api.")


def download_audio_stage2(url: str) -> Tuple[Path, Dict]:
    TMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio[ext=m4a]/bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "outtmpl": str(TMP_AUDIO_DIR / "%(id)s.%(ext)s"),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
                "preferredquality": "32",
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id")
        if not video_id:
            raise RuntimeError("yt-dlp did not return a video id.")

    files = sorted(TMP_AUDIO_DIR.glob(f"{video_id}*.m4a"))
    if not files:
        files = sorted(TMP_AUDIO_DIR.glob(f"{video_id}*.*"))
    if not files:
        raise RuntimeError("Audio file was not created by yt-dlp.")
    return files[0], info


def cleanup_audio_files(audio_path: Optional[Path], video_id: str) -> None:
    if audio_path and audio_path.exists():
        try:
            audio_path.unlink()
        except Exception:
            pass
    for leftover in TMP_AUDIO_DIR.glob(f"{video_id}*.*"):
        if leftover.exists():
            try:
                leftover.unlink()
            except Exception:
                pass


def transcribe_with_groq(audio_path: Path, groq_api_key: str, model_name: str) -> str:
    client = Groq(api_key=groq_api_key)
    audio_bytes = audio_path.read_bytes()
    base_kwargs = {
        "model": model_name,
        "file": (audio_path.name, audio_bytes),
        "temperature": 0,
        "response_format": "json",
    }
    arabic_prompt = "Detect spoken language. If Arabic, transcribe in Arabic script."
    try:
        result = client.audio.transcriptions.create(prompt=arabic_prompt, **base_kwargs)
    except Exception:
        result = client.audio.transcriptions.create(**base_kwargs)
    finally:
        del audio_bytes

    text = result.text if hasattr(result, "text") else result.get("text", "")
    if not text or not str(text).strip():
        raise RuntimeError("Groq returned empty transcription.")
    return str(text).strip()


def transcribe_with_hf(audio_path: Path, hf_api_key: str, model_name: str, max_retries: int = 3) -> str:
    endpoint = f"https://api-inference.huggingface.co/models/{quote(model_name, safe='')}"
    headers = {"Authorization": f"Bearer {hf_api_key}", "Content-Type": "audio/m4a"}
    audio_bytes = audio_path.read_bytes()

    try:
        for attempt in range(max_retries):
            req = urllib.request.Request(endpoint, data=audio_bytes, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=300) as response:
                    raw = response.read().decode("utf-8", errors="ignore")
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="ignore")

            payload = json.loads(raw) if raw.strip().startswith(("{", "[")) else {"error": raw}
            if isinstance(payload, dict) and payload.get("error"):
                err = str(payload.get("error"))
                if "loading" in err.lower() and attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                raise RuntimeError(err)

            if isinstance(payload, dict):
                text = str(payload.get("text", "")).strip()
                if text: return text
            elif isinstance(payload, list):
                joined = " ".join(str(item.get("text", "")).strip() for item in payload if isinstance(item, dict)).strip()
                if joined: return joined

            if attempt < max_retries - 1: time.sleep(3)
        raise RuntimeError("HF returned empty transcription.")
    finally:
        del audio_bytes


def transcribe_with_gemini_audio(audio_path: Path, gemini_api_key: str, model_name: str) -> str:
    genai.configure(api_key=gemini_api_key)
    uploaded = genai.upload_file(path=str(audio_path))
    prompt = "Transcribe the audio accurately. Detect Arabic language if present."
    try:
        model = GenerativeModel(model_name=model_name)
        response = model.generate_content([prompt, uploaded])
        text = (response.text or "").strip()
        if not text: raise RuntimeError("Gemini empty response.")
        return text
    finally:
        try: genai.delete_file(uploaded.name)
        except Exception: pass


def process_single_url(url: str, groq_api_key: str, hf_api_key: str, gemini_api_key: str, groq_model: str, hf_model: str, gemini_model: str) -> Dict:
    started = utc_now_iso()
    video_id = extract_video_id(url)
    record = {
        "url": url, "video_id": video_id or "", "title": "", "status": "failed",
        "method_used": "", "transcript": "", "error": "", "started_at_utc": started, "timestamp_utc": started,
    }
    if not video_id:
        record["error"] = "Invalid URL."
        return record

    errors: List[str] = []
    
    # Stage 1: Official Transcript
    try:
        text = fetch_transcript_stage1(video_id)
        record.update({"status": "success", "method_used": "youtube-transcript-api", "transcript": text, "timestamp_utc": utc_now_iso()})
        return record
    except Exception as exc:
        errors.append(f"Stage1: {exc}")

    # Stage 2: Audio Download
    audio_path = None
    try:
        audio_path, info = download_audio_stage2(url)
        record["title"] = info.get("title", "")
    except Exception as exc:
        errors.append(f"Stage2: {exc}")
        record["error"] = " | ".join(errors)
        return record

    # Stage 3: AI STT
    try:
        # Try Groq
        if groq_api_key:
            try:
                text = transcribe_with_groq(audio_path, groq_api_key, groq_model)
                record.update({"status": "success", "method_used": "groq", "transcript": text, "timestamp_utc": utc_now_iso()})
                return record
            except Exception as e: errors.append(f"Groq: {e}")
        
        # Try HF
        if hf_api_key:
            try:
                text = transcribe_with_hf(audio_path, hf_api_key, hf_model)
                record.update({"status": "success", "method_used": "huggingface", "transcript": text, "timestamp_utc": utc_now_iso()})
                return record
            except Exception as e: errors.append(f"HF: {e}")

        # Try Gemini
        if gemini_api_key:
            try:
                text = transcribe_with_gemini_audio(audio_path, gemini_api_key, gemini_model)
                record.update({"status": "success", "method_used": "gemini-audio", "transcript": text, "timestamp_utc": utc_now_iso()})
                return record
            except Exception as e: errors.append(f"Gemini: {e}")
    finally:
        cleanup_audio_files(audio_path, video_id)

    record["error"] = " | ".join(errors)
    return record


def get_dashboard_stats(all_urls: List[str], records: Dict[str, Dict]) -> Dict[str, float]:
    total = len(all_urls)
    processed = 0
    success = 0
    failed = 0
    for url in all_urls:
        rec = records.get(url)
        if not rec: continue
        processed += 1
        if rec.get("status") == "success": success += 1
        elif rec.get("status") == "failed": failed += 1
    return {"total": total, "processed": processed, "success": success, "failed": failed, "success_rate": (success/processed*100 if processed else 0.0)}


def ordered_resume_indices(all_urls: List[str], records: Dict[str, Dict], last_processed_index: int, retry_failed: bool) -> List[int]:
    indices = []
    for idx, url in enumerate(all_urls):
        rec = records.get(url)
        if rec is None or (retry_failed and rec.get("status") == "failed"):
            indices.append(idx)
    if not indices: return []
    start = (last_processed_index + 1) % len(all_urls) if last_processed_index >= 0 else 0
    return sorted(indices, key=lambda idx: (idx - start) % len(all_urls))


def render_live_table(placeholder, all_urls: List[str], records: Dict[str, Dict]) -> None:
    rows = []
    for url in all_urls:
        rec = records.get(url, {})
        rows.append({"URL": url, "Status": rec.get("status", "pending"), "Method": rec.get("method_used", ""), "Time": rec.get("timestamp_utc", "")})
    placeholder.dataframe(pd.DataFrame(rows), use_container_width=True, height=340)


def build_markdown_report(success_records: List[Dict]) -> str:
    lines = ["# YouTube Bulk Transcript Report", f"Generated: {utc_now_iso()}", ""]
    for idx, rec in enumerate(success_records, start=1):
        lines.extend([f"## {idx}. {rec.get('title', 'Untitled')}", f"- URL: {rec['url']}", f"- Method: {rec['method_used']}", "", rec['transcript'], ""])
    return "\n".join(lines)


def markdown_to_pdf_bytes(markdown_text: str) -> Tuple[Optional[bytes], str]:
    try:
        from fpdf import FPDF
    except: return None, "Install fpdf2."
    if not ARABIC_FONT_FILE.exists(): return None, "Missing Font."
    pdf = FPDF(); pdf.add_page()
    try: pdf.set_text_shaping(True)
    except: pass
    pdf.add_font("Amiri", "", str(ARABIC_FONT_FILE)); pdf.set_font("Amiri", size=13)
    for line in markdown_text.splitlines():
        line = re.sub(r"^#{1,6}\s*", "", line).replace("**", "").replace("`", "") or " "
        align = "R" if bool(re.search(r"[\u0600-\u06FF]", line)) else "L"
        pdf.multi_cell(0, 7, line, align=align)
    return bytes(pdf.output(dest="S")), ""


def simple_query_chunks(records: List[Dict], question: str) -> Tuple[str, List[str]]:
    terms = set(re.findall(r"[\w\u0600-\u06FF]{3,}", question.lower()))
    scored = []
    for rec in records:
        text = rec.get("transcript", "")
        for i in range(0, len(text), 1400):
            chunk = text[i:i+1700]
            score = len(terms.intersection(set(re.findall(r"[\w\u0600-\u06FF]{3,}", chunk.lower())))) if terms else 0
            scored.append((score, chunk, rec['url'], rec.get('title', 'Video')))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:12]
    return "\n".join(f"[Source: {t} | {u}]\n{c}" for _, c, u, t in top)[:60000], list(dict.fromkeys(u for _, _, u, _ in top))


def ask_gemini_on_context(question: str, context: str, gemini_api_key: str, model_name: str) -> str:
    genai.configure(api_key=gemini_api_key)
    model = GenerativeModel(model_name=model_name)
    prompt = f"Answer based ONLY on context:\nQuestion: {question}\n\nContext: {context}"
    return (model.generate_content(prompt).text or "").strip()


def process_urls_ui(target_indices, all_urls, records, groq_api_key, hf_api_key, gemini_api_key, groq_model, hf_model, gemini_model, live_table_placeholder):
    total = len(target_indices)
    progress = st.progress(0.0)
    status_box = st.empty()
    success_count = 0
    
    for idx, url_idx in enumerate(target_indices, start=1):
        url = all_urls[url_idx]
        record = process_single_url(url, groq_api_key, hf_api_key, gemini_api_key, groq_model, hf_model, gemini_model)
        records[url] = record
        st.session_state.last_processed_index = url_idx
        save_progress(all_urls, records, url_idx)
        if record["status"] == "success": success_count += 1
        progress.progress(idx / total)
        status_box.info(f"Progress: {idx}/{total} | Successful: {success_count}")
        render_live_table(live_table_placeholder, all_urls, records)
        if idx % 10 == 0: gc.collect()
        if idx < total: time.sleep(RATE_LIMIT_SECONDS)
    st.success("Run completed.")


def main():
    st.set_page_config(page_title="Bulk YT Intel", layout="wide")
    st.title("Bulk YouTube Content Intelligence")
    init_state()

    g_api = st.secrets.get("GROQ_API_KEY", "")
    h_api = st.secrets.get("HF_API_KEY", "")
    gem_api = st.secrets.get("GEMINI_API_KEY", "")

    with st.sidebar:
        st.subheader("Settings")
        st.write(f"Groq: {'✅' if g_api else '❌'}")
        st.write(f"HF: {'✅' if h_api else '❌'}")
        st.write(f"Gemini: {'✅' if gem_api else '❌'}")
        g_mod = st.text_input("Groq Model", "whisper-large-v3")
        h_mod = st.text_input("HF Model", "openai/whisper-large-v3")
        gem_mod = st.text_input("Gemini Model", "gemini-1.5-flash")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Input URLs")
        url_text = st.text_area("Paste URLs", height=200)
        b1, b2 = st.columns(2)
        if b1.button("Replace List"):
            urls = parse_bulk_urls(url_text)
            st.session_state.records = {u: st.session_state.records[u] for u in urls if u in st.session_state.records}
            st.session_state.all_urls = urls
            st.session_state.last_processed_index = infer_last_processed_index(urls, st.session_state.records)
            save_progress(urls, st.session_state.records, st.session_state.last_processed_index)
            st.rerun()
        if b2.button("Append"):
            urls = parse_bulk_urls(url_text)
            merged = list(dict.fromkeys(st.session_state.all_urls + urls))
            st.session_state.all_urls = merged
            save_progress(merged, st.session_state.records, st.session_state.last_processed_index)
            st.rerun()

    with col2:
        st.subheader("Status")
        st.write(f"Current Index: {st.session_state.last_processed_index}")
        if st.button("Reload from Disk"):
            st.session_state.initialized = False
            st.rerun()

    st.divider()
    stats = get_dashboard_stats(st.session_state.all_urls, st.session_state.records)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total", int(stats["total"]))
    m2.metric("Processed", int(stats["processed"]))
    m3.metric("Success", int(stats["success"]))
    m4.metric("Rate", f"{stats['success_rate']:.1f}%")

    table_p = st.empty()
    render_live_table(table_p, st.session_state.all_urls, st.session_state.records)

    st.subheader("Controls")
    retry = st.checkbox("Retry Failed")
    indices = ordered_resume_indices(st.session_state.all_urls, st.session_state.records, st.session_state.last_processed_index, retry)
    st.write(f"Pending: {len(indices)}")

    p1, p2 = st.columns(2)
    if p1.button(f"Process Batch ({BATCH_SIZE})"):
        process_urls_ui(indices[:BATCH_SIZE], st.session_state.all_urls, st.session_state.records, g_api, h_api, gem_api, g_mod, h_mod, gem_mod, table_p)
    if p2.button("Process All"):
        process_urls_ui(indices, st.session_state.all_urls, st.session_state.records, g_api, h_api, gem_api, g_mod, h_mod, gem_mod, table_p)

    st.divider()
    success_recs = [r for r in st.session_state.records.values() if r.get("status") == "success"]
    if success_recs:
        report = build_markdown_report(success_recs)
        st.download_button("Download MD", report.encode("utf-8"), "report.md")
        pdf, err = markdown_to_pdf_bytes(report)
        if pdf: st.download_button("Download PDF", pdf, "report.pdf")
        else: st.warning(err)
        
        st.subheader("Chat with Videos")
        q = st.text_input("Question")
        if st.button("Ask Gemini") and q:
            ctx, srcs = simple_query_chunks(success_recs, q)
            ans = ask_gemini_on_context(q, ctx, gem_api, gem_mod)
            st.session_state.chat_history.append({"q": q, "a": ans, "s": srcs})
            for chat in reversed(st.session_state.chat_history):
                st.write(f"**Q:** {chat['q']}\n\n**A:** {chat['a']}")
                st.caption(f"Sources: {chat['s']}")
                st.divider()


if __name__ == "__main__":
    main()
