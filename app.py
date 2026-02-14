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
        data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {"all_urls": [], "records": {}, "last_processed_index": -1}

    all_urls = data.get("all_urls", [])
    raw_records = data.get("records", [])
    records_map: Dict[str, Dict] = {}

    if isinstance(raw_records, dict):
        for rec in raw_records.values():
            if isinstance(rec, dict) and rec.get("url"):
                records_map[rec["url"]] = rec
    elif isinstance(raw_records, list):
        for rec in raw_records:
            if isinstance(rec, dict) and rec.get("url"):
                records_map[rec["url"]] = rec

    raw_last_index = data.get("last_processed_index", -1)
    if isinstance(raw_last_index, int):
        last_processed_index = raw_last_index
    else:
        last_processed_index = infer_last_processed_index(all_urls, records_map)

    if all_urls:
        last_processed_index = min(last_processed_index, len(all_urls) - 1)
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
    # Compatibility across youtube-transcript-api versions.
    try:
        segments = YouTubeTranscriptApi.get_transcript(video_id, languages=["ar", "en", "en-US"])
        text = " ".join(seg.get("text", "").strip() for seg in segments if seg.get("text"))
        if text.strip():
            return text.strip()
    except AttributeError:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id)
        if hasattr(fetched, "to_raw_data"):
            segments = fetched.to_raw_data()
            text = " ".join(seg.get("text", "").strip() for seg in segments if seg.get("text"))
            if text.strip():
                return text.strip()
    except Exception:
        pass

    segments = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join(seg.get("text", "").strip() for seg in segments if seg.get("text"))
    if not text.strip():
        raise RuntimeError("Empty transcript from youtube-transcript-api.")
    return text.strip()


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
    arabic_prompt = (
        "Detect spoken language automatically. "
        "If Arabic speech is present, transcribe it accurately in Arabic script. "
        "Do not translate."
    )
    try:
        result = client.audio.transcriptions.create(prompt=arabic_prompt, **base_kwargs)
    except TypeError:
        result = client.audio.transcriptions.create(**base_kwargs)
    finally:
        del audio_bytes

    text = result.text if hasattr(result, "text") else result.get("text", "")
    if not text or not str(text).strip():
        raise RuntimeError("Groq returned empty transcription.")
    return str(text).strip()


def transcribe_with_hf(audio_path: Path, hf_api_key: str, model_name: str, max_retries: int = 3) -> str:
    if not hf_api_key:
        raise RuntimeError("HF_API_KEY is missing.")
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
                if text:
                    return text
            elif isinstance(payload, list):
                joined = " ".join(
                    str(item.get("text", "")).strip() for item in payload if isinstance(item, dict)
                ).strip()
                if joined:
                    return joined

            if attempt < max_retries - 1:
                time.sleep(3)
        raise RuntimeError("Hugging Face returned empty transcription.")
    finally:
        del audio_bytes


def transcribe_with_gemini_audio(audio_path: Path, gemini_api_key: str, model_name: str) -> str:
    genai.configure(api_key=gemini_api_key)
    uploaded = genai.upload_file(path=str(audio_path))
    prompt = (
        "You are an accurate transcription engine. "
        "Detect spoken language automatically. "
        "If Arabic speech exists, transcribe it precisely in Arabic script. "
        "Do not translate and do not summarize. Return plain text only."
    )
    try:
        model = GenerativeModel(model_name=model_name)
        response = model.generate_content([prompt, uploaded])
        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned empty transcription.")
        return text
    finally:
        try:
            genai.delete_file(uploaded.name)
        except Exception:
            pass


def process_single_url(
    url: str,
    groq_api_key: str,
    hf_api_key: str,
    gemini_api_key: str,
    groq_model: str,
    hf_model: str,
    gemini_model: str,
) -> Dict:
    started = utc_now_iso()
    video_id = extract_video_id(url)
    record = {
        "url": url,
        "video_id": video_id or "",
        "title": "",
        "status": "failed",
        "method_used": "",
        "transcript": "",
        "error": "",
        "started_at_utc": started,
        "timestamp_utc": started,
    }
    if not video_id:
        record["error"] = "Invalid YouTube URL or video id."
        return record

    errors: List[str] = []
    audio_path: Optional[Path] = None
    title = ""

    try:
        text = fetch_transcript_stage1(video_id)
        record.update(
            {
                "status": "success",
                "method_used": "youtube-transcript-api",
                "transcript": text,
                "timestamp_utc": utc_now_iso(),
            }
        )
        return record
    except Exception as exc:
        errors.append(f"stage1: {exc}")

    try:
        audio_path, info = download_audio_stage2(url)
        title = info.get("title", "") if isinstance(info, dict) else ""
        record["title"] = title
    except Exception as exc:
        errors.append(f"stage2: {exc}")
        record["error"] = " | ".join(errors)
        record["timestamp_utc"] = utc_now_iso()
        return record

    try:
        try:
            if groq_api_key:
                text = transcribe_with_groq(audio_path, groq_api_key, groq_model)
                record.update(
                    {
                        "status": "success",
                        "method_used": "groq",
                        "transcript": text,
                        "timestamp_utc": utc_now_iso(),
                    }
                )
                return record
            errors.append("stage3/groq: missing GROQ_API_KEY")
        except Exception as exc:
            errors.append(f"stage3/groq: {exc}")

        try:
            if hf_api_key:
                text = transcribe_with_hf(audio_path, hf_api_key, hf_model)
                record.update(
                    {
                        "status": "success",
                        "method_used": "huggingface-api",
                        "transcript": text,
                        "timestamp_utc": utc_now_iso(),
                    }
                )
                return record
            errors.append("stage3/hf: missing HF_API_KEY")
        except Exception as exc:
            errors.append(f"stage3/hf: {exc}")

        try:
            if gemini_api_key:
                text = transcribe_with_gemini_audio(audio_path, gemini_api_key, gemini_model)
                record.update(
                    {
                        "status": "success",
                        "method_used": "gemini-audio",
                        "transcript": text,
                        "timestamp_utc": utc_now_iso(),
                    }
                )
                return record
            errors.append("stage3/gemini: missing GEMINI_API_KEY")
        except Exception as exc:
            errors.append(f"stage3/gemini: {exc}")
    finally:
        cleanup_audio_files(audio_path, video_id)

    record["error"] = " | ".join(errors)
    record["timestamp_utc"] = utc_now_iso()
    return record


def get_dashboard_stats(all_urls: List[str], records: Dict[str, Dict]) -> Dict[str, float]:
    total = len(all_urls)
    processed = 0
    success = 0
    failed = 0
    for url in all_urls:
        rec = records.get(url)
        if not rec:
            continue
        processed += 1
        if rec.get("status") == "success":
            success += 1
        elif rec.get("status") == "failed":
            failed += 1
    success_rate = (success / processed * 100.0) if processed else 0.0
    return {
        "total": total,
        "processed": processed,
        "success": success,
        "failed": failed,
        "success_rate": success_rate,
    }


def pending_indices(all_urls: List[str], records: Dict[str, Dict], retry_failed: bool) -> List[int]:
    indices = []
    for idx, url in enumerate(all_urls):
        rec = records.get(url)
        if rec is None:
            indices.append(idx)
        elif retry_failed and rec.get("status") == "failed":
            indices.append(idx)
    return indices


def ordered_resume_indices(
    all_urls: List[str],
    records: Dict[str, Dict],
    last_processed_index: int,
    retry_failed: bool,
) -> List[int]:
    if not all_urls:
        return []
    candidates = pending_indices(all_urls, records, retry_failed=retry_failed)
    if not candidates:
        return []
    start = (last_processed_index + 1) % len(all_urls) if last_processed_index >= 0 else 0
    return sorted(candidates, key=lambda idx: (idx - start) % len(all_urls))


def build_live_dataframe(all_urls: List[str], records: Dict[str, Dict]) -> pd.DataFrame:
    rows = []
    for url in all_urls:
        rec = records.get(url, {})
        rows.append(
            {
                "URL": url,
                "Status": rec.get("status", "pending"),
                "Method Used": rec.get("method_used", ""),
                "Timestamp": rec.get("timestamp_utc", ""),
            }
        )
    return pd.DataFrame(rows, columns=["URL", "Status", "Method Used", "Timestamp"])


def render_live_table(placeholder, all_urls: List[str], records: Dict[str, Dict]) -> None:
    df = build_live_dataframe(all_urls, records)
    placeholder.dataframe(df, use_container_width=True, height=340)


def build_markdown_report(success_records: List[Dict]) -> str:
    lines = [
        "# YouTube Bulk Transcript Report",
        "",
        f"Generated UTC: {utc_now_iso()}",
        f"Total successful videos: {len(success_records)}",
        "",
    ]
    for idx, rec in enumerate(success_records, start=1):
        title = rec.get("title") or rec.get("video_id") or "Untitled"
        lines.append(f"## {idx}. {title}")
        lines.append(f"- URL: {rec.get('url', '')}")
        lines.append(f"- Method: {rec.get('method_used', '')}")
        lines.append("")
        lines.append(rec.get("transcript", ""))
        lines.append("")
    return "\n".join(lines)


def contains_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def markdown_to_pdf_bytes(markdown_text: str) -> Tuple[Optional[bytes], str]:
    try:
        from fpdf import FPDF
    except Exception:
        return None, "Install `fpdf2` to enable PDF export."

    if not ARABIC_FONT_FILE.exists():
        return None, "Missing `Amiri-Regular.ttf` in project root. Arabic PDF export requires this font."

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    try:
        pdf.set_text_shaping(True)
    except Exception:
        pass
    pdf.add_font("Amiri", "", str(ARABIC_FONT_FILE))
    pdf.set_font("Amiri", size=13)

    for raw_line in markdown_text.splitlines():
        line = re.sub(r"^#{1,6}\s*", "", raw_line)
        line = line.replace("**", "").replace("`", "")
        line = line if line.strip() else " "
        align = "R" if contains_arabic(line) else "L"
        pdf.multi_cell(0, 7, line, align=align)

    result = pdf.output(dest="S")
    if isinstance(result, (bytes, bytearray)):
        return bytes(result), ""
    return result.encode("latin-1"), ""


def simple_query_chunks(records: List[Dict], question: str, max_chunks: int = 12) -> Tuple[str, List[str]]:
    terms = set(re.findall(r"[a-zA-Z0-9\u0600-\u06FF]{3,}", question.lower()))
    scored = []

    for rec in records:
        text = rec.get("transcript", "")
        if not text:
            continue
        url = rec.get("url", "")
        title = rec.get("title") or rec.get("video_id") or "Untitled"
        chunk_size = 1700
        step = 1400
        for i in range(0, len(text), step):
            chunk = text[i : i + chunk_size]
            if not chunk:
                continue
            tokens = set(re.findall(r"[a-zA-Z0-9\u0600-\u06FF]{3,}", chunk.lower()))
            score = len(terms.intersection(tokens)) if terms else 0
            scored.append((score, chunk, url, title))

    if not scored:
        return "", []

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max_chunks]
    context_parts = []
    sources = []
    for _, chunk, url, title in top:
        context_parts.append(f"[Source: {title} | {url}]\n{chunk}\n")
        sources.append(url)
    return "\n".join(context_parts)[:60000], list(dict.fromkeys(sources))


def ask_gemini_on_context(question: str, context: str, gemini_api_key: str, model_name: str) -> str:
    if not gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is missing in Streamlit secrets.")
    if not context.strip():
        raise RuntimeError("No transcript context available.")

    genai.configure(api_key=gemini_api_key)
    model = GenerativeModel(model_name=model_name)
    prompt = (
        "Answer the question using only the provided transcript context. "
        "If context is insufficient, say what is missing.\n\n"
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n"
    )
    response = model.generate_content(prompt)
    return (response.text or "").strip()


def process_urls_ui(
    target_indices: List[int],
    all_urls: List[str],
    records: Dict[str, Dict],
    groq_api_key: str,
    hf_api_key: str,
    gemini_api_key: str,
    groq_model: str,
    hf_model: str,
    gemini_model: str,
    live_table_placeholder,
) -> None:
    total = len(target_indices)
    if total == 0:
        st.info("No URLs to process.")
        return

    progress = st.progress(0.0)
    status_box = st.empty()
    success_count = 0
    failed_count = 0

    for idx, url_index in enumerate(target_indices, start=1):
        url = all_urls[url_index]
        record = process_single_url(
            url=url,
            groq_api_key=groq_api_key,
            hf_api_key=hf_api_key,
            gemini_api_key=gemini_api_key,
            groq_model=groq_model,
            hf_model=hf_model,
            gemini_model=gemini_model,
        )
        records[url] = record
        st.session_state.last_processed_index = url_index
        save_progress(all_urls, records, st.session_state.last_processed_index)

        if record.get("status") == "success":
            success_count += 1
        else:
            failed_count += 1

        progress.progress(idx / total)
        status_box.info(
            f"Processed {idx}/{total} this run | Success: {success_count} | Failed: {failed_count}"
        )
        render_live_table(live_table_placeholder, all_urls, records)

        if idx % 10 == 0:
            gc.collect()
        if idx < total:
            time.sleep(RATE_LIMIT_SECONDS)

    st.session_state.records = records
    st.success(f"Run completed. Success={success_count}, Failed={failed_count}")


def main() -> None:
    st.set_page_config(page_title="Bulk YouTube Intelligence", layout="wide")
    st.title("Bulk YouTube Content Intelligence")
    st.caption("Resumable 1,000-URL processing with fail-safe extraction and Arabic-ready exports.")

    init_state()

    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = ""
    try:
        hf_api_key = st.secrets["HF_API_KEY"]
    except Exception:
        hf_api_key = ""
    try:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        gemini_api_key = ""

    with st.sidebar:
        st.subheader("Secrets Status")
        st.write(f"GROQ_API_KEY: {'Loaded' if groq_api_key else 'Missing'}")
        st.write(f"HF_API_KEY: {'Loaded' if hf_api_key else 'Missing'}")
        st.write(f"GEMINI_API_KEY: {'Loaded' if gemini_api_key else 'Missing'}")
        st.subheader("Model Settings")
        groq_model = st.text_input("Groq STT model", value="whisper-large-v3")
        hf_model = st.text_input("HF STT model", value="openai/whisper-large-v3")
        gemini_model = st.text_input("Gemini model", value="gemini-1.5-flash")
        st.caption("API keys are read from Streamlit secrets only.")

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Bulk Input")
        url_text = st.text_area(
            "Paste up to 1000 YouTube URLs (one per line or comma-separated)",
            height=250,
            placeholder="https://www.youtube.com/watch?v=...\nhttps://youtu.be/...",
        )
        b1, b2 = st.columns(2)
        if b1.button("Load / Replace URL List", use_container_width=True):
            urls = parse_bulk_urls(url_text)
            kept_records = {url: st.session_state.records[url] for url in urls if url in st.session_state.records}
            st.session_state.all_urls = urls
            st.session_state.records = kept_records
            st.session_state.last_processed_index = infer_last_processed_index(urls, kept_records)
            save_progress(urls, kept_records, st.session_state.last_processed_index)
            st.success(f"Loaded {len(urls)} normalized URLs.")

        if b2.button("Append URLs", use_container_width=True):
            urls = parse_bulk_urls(url_text)
            merged = list(dict.fromkeys(st.session_state.all_urls + urls))
            st.session_state.all_urls = merged
            st.session_state.last_processed_index = infer_last_processed_index(merged, st.session_state.records)
            save_progress(merged, st.session_state.records, st.session_state.last_processed_index)
            st.success(f"Added {len(urls)} URLs. Total tracked: {len(merged)}")

    with col2:
        st.subheader("Storage")
        st.write(f"`{DATA_FILE.resolve()}`")
        st.write(f"`{CSV_FILE.resolve()}`")
        if st.button("Reload Progress From Disk", use_container_width=True):
            payload = load_progress()
            st.session_state.all_urls = payload.get("all_urls", [])
            st.session_state.records = payload.get("records", {})
            st.session_state.last_processed_index = payload.get("last_processed_index", -1)
            st.success("Progress reloaded from disk.")

        next_resume_index = st.session_state.last_processed_index + 1
        st.info(f"Last processed index: {st.session_state.last_processed_index}")
        st.info(f"Resume index: {next_resume_index}")

    st.divider()
    stats = get_dashboard_stats(st.session_state.all_urls, st.session_state.records)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total URLs", int(stats["total"]))
    m2.metric("Processed", int(stats["processed"]))
    m3.metric("Failed", int(stats["failed"]))
    m4.metric("Success Rate", f"{stats['success_rate']:.2f}%")

    st.subheader("Live Processing Table")
    live_table_placeholder = st.empty()
    render_live_table(live_table_placeholder, st.session_state.all_urls, st.session_state.records)

    st.subheader("Processing Controls")
    retry_failed = st.checkbox("Include failed URLs in resume run", value=False)
    resume_indices = ordered_resume_indices(
        st.session_state.all_urls,
        st.session_state.records,
        st.session_state.last_processed_index,
        retry_failed=retry_failed,
    )
    st.write(f"Pending URLs in resume order: **{len(resume_indices)}**")

    p1, p2 = st.columns(2)
    if p1.button(f"Resume Next Batch ({BATCH_SIZE})", use_container_width=True):
        if not st.session_state.all_urls:
            st.warning("Load URLs first.")
        elif not resume_indices:
            st.info("No pending URLs to process.")
        else:
            process_urls_ui(
                target_indices=resume_indices[:BATCH_SIZE],
                all_urls=st.session_state.all_urls,
                records=st.session_state.records,
                groq_api_key=groq_api_key,
                hf_api_key=hf_api_key,
                gemini_api_key=gemini_api_key,
                groq_model=groq_model,
                hf_model=hf_model,
                gemini_model=gemini_model,
                live_table_placeholder=live_table_placeholder,
            )

    if p2.button("Resume All Remaining", use_container_width=True):
        if not st.session_state.all_urls:
            st.warning("Load URLs first.")
        elif not resume_indices:
            st.info("No pending URLs to process.")
        else:
            process_urls_ui(
                target_indices=resume_indices,
                all_urls=st.session_state.all_urls,
                records=st.session_state.records,
                groq_api_key=groq_api_key,
                hf_api_key=hf_api_key,
                gemini_api_key=gemini_api_key,
                groq_model=groq_model,
                hf_model=hf_model,
                gemini_model=gemini_model,
                live_table_placeholder=live_table_placeholder,
            )

    st.divider()
    st.subheader("Download Center")
    success_records = [
        rec for rec in st.session_state.records.values() if rec.get("status") == "success" and rec.get("transcript")
    ]
    if success_records:
        markdown_report = build_markdown_report(success_records)
        st.download_button(
            label="Download Markdown Report",
            data=markdown_report.encode("utf-8"),
            file_name="youtube_bulk_report.md",
            mime="text/markdown",
            use_container_width=True,
        )
        pdf_bytes, pdf_error = markdown_to_pdf_bytes(markdown_report)
        if pdf_bytes:
            st.download_button(
                label="Download PDF Report (Arabic Ready)",
                data=pdf_bytes,
                file_name="youtube_bulk_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        else:
            st.warning(pdf_error)
    else:
        st.info("No successful transcripts available yet.")

    st.divider()
    st.subheader("Ask Gemini About Processed Videos")
    question = st.text_input("Ask a question over all extracted transcripts")
    if st.button("Ask", use_container_width=True):
        if not question.strip():
            st.warning("Enter a question first.")
        elif not success_records:
            st.warning("No successful transcripts available yet.")
        else:
            context, sources = simple_query_chunks(success_records, question)
            try:
                answer = ask_gemini_on_context(
                    question=question,
                    context=context,
                    gemini_api_key=gemini_api_key,
                    model_name=gemini_model,
                )
                st.session_state.chat_history.append(
                    {"question": question, "answer": answer, "sources": sources, "at": utc_now_iso()}
                )
            except Exception as exc:
                st.error(f"Gemini query failed: {exc}")

    if st.session_state.chat_history:
        for item in reversed(st.session_state.chat_history[-10:]):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer']}")
            if item.get("sources"):
                st.caption("Sources: " + ", ".join(item["sources"][:8]))
            st.markdown("---")


if __name__ == "__main__":
    main()
