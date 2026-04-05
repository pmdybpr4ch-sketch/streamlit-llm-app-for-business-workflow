import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# ==========================================
# 環境変数の読み込み
# ==========================================
# ローカル開発時は .env を使用

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "OPENAI_API_KEY が設定されていません。`.env` ファイルに `OPENAI_API_KEY=<your-key>` を追加してください。"
    )
    st.stop()

# ==========================================
# 定数定義
# ==========================================
MAX_TOKENS_PER_SESSION = 10000  # 1セッションあたりのトークン上限（約15往復分）
MAX_TURNS = 10                  # 会話履歴として送る最大ターン数（コスト抑制）

# ==========================================
# システムプロンプトの定義
# ==========================================
system_prompt_analysis = """
あなたはビジネスプロセス改善に特化した「業務分析スペシャリスト」です。
さまざまな業界の業務ワークフロー（営業、マーケティング、顧客対応、バックオフィス、プロジェクト管理、製造・運用、情報共有など）に精通し、非効率の原因分析を得意としています。

**主な役割**
- ユーザーが語る業務プロセスを丁寧に聞き出し、フローを整理・可視化する
- 属人化、ボトルネック、重複作業、非効率な情報連携、意思決定の遅れなどの課題を鋭く指摘する
- 課題の背景や影響を論理的に説明し、優先順位付けを行う

**厳格な回答方針**
- 課題の分析と整理に徹してください。具体的なSaaSツール名、生成AIツール、解決策の提案は一切行わない。
- 「このようなプロセスではよく...」「属人性が生まれる典型的な理由は...」のように、一般的なビジネス知見を基にユーザーの状況に寄り添った分析をする。
- 必要に応じて深掘り質問をし、業務の全体像を明確に整理する。
- 回答はプロフェッショナルで客観的、現場の痛みを理解した建設的なトーンを保つ。

**禁止事項**
- ツール提案や「こうすれば解決します」という解決策に踏み込まない。
- 分析が浅くならないよう、具体例を交えつつ一般化しすぎない。

ユーザーの相談に対して、まずは「現状の業務を整理し、隠れた非効率やリスクを明らかにする」ことに集中してください。
"""

system_prompt_solution = """
あなたはビジネス効率化に詳しい「生成AI・SaaS活用コンサルタント」です。
さまざまな業界の業務課題に対して、最新のSaaSツールと生成AIを組み合わせた現実的なソリューションを提案する専門家です。

**主な役割**
- ユーザーの業務課題に対して、生成AIとSaaSを活用した即効性・費用対効果の高い解決策を提案する
- 特に強い領域：業務自動化、情報共有・ナレッジ管理、顧客対応効率化、プロジェクト・タスク管理、データ分析・レポート作成、クリエイティブ作業支援、レビュー・承認プロセスなど

**回答方針**
- 提案は2025-2026年時点で実用レベルにあるツール・手法を中心に（例：Notion/Airtable/Codaなどのノーコードツール、Zapier/Makeなどの自動化、ChatGPT/Claude/Geminiなどの生成AI活用、Slack/Teams連携、専用SaaSなど）。
- 各提案で以下の点を明確に説明：どの業務課題に対して有効か / 期待できる効果（時間短縮、コスト削減、ミス低減など）/ 導入時の現実的なステップと注意点（データセキュリティ、学習コスト、既存システムとの連携、人間による最終確認など）
- 「補助ツール」として活用し、人間の判断や創造性を置き換えないよう強調する。
- 提案は段階的（低コスト・低リスクから始められるもの優先）にし、ユーザーの規模や予算感に配慮した現実的な内容にする。
- ツールの組み合わせ（例：生成AI+ノーコードツール）を積極的に提案。

**禁止事項**
- 課題分析だけに留まらない（課題はすでに把握している前提で提案に集中）。
- 根拠のない過度な期待や「万能ツール」のような提案は避け、失敗しがちなポイントも正直に伝える。

回答は具体的で実践的、「すぐに試せる」レベルを目指してください。
"""

# ==========================================
# session_state の初期化
# ==========================================
if "llm" not in st.session_state:
    st.session_state.llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.5)

if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role": "user"/"assistant", "content": "..."}

if "total_tokens" not in st.session_state:
    st.session_state.total_tokens = 0

if "current_expert" not in st.session_state:
    st.session_state.current_expert = None

# ==========================================
# 関数定義
# ==========================================
def get_llm_response(input_text: str, expert_type: str) -> str:
    """
    会話履歴を踏まえた応答を返す（LCEL スタイル）。
    - 専門家モードが切り替わった場合は履歴をリセット
    - トークン上限に達した場合は警告して停止
    - 直近 MAX_TURNS ターンのみを送信してコストを抑制
    """

    # トークン上限チェック
    if st.session_state.total_tokens >= MAX_TOKENS_PER_SESSION:
        st.warning(
            "本セッションの利用上限に達しました。"
            "ページを再読み込みすると新しいセッションが始まります。"
        )
        st.stop()

    # 専門家モードが切り替わったら履歴をリセット
    if st.session_state.current_expert != expert_type:
        st.session_state.messages = []
        st.session_state.current_expert = expert_type

    # システムプロンプトを選択
    system_message = (
        system_prompt_analysis
        if expert_type == "業務分析スペシャリスト（課題発見・整理）"
        else system_prompt_solution
    )

    # 直近 MAX_TURNS ターン分だけを履歴として使用
    recent = st.session_state.messages[-(MAX_TURNS * 2):]

    # LangChain Core の message 形式に変換
    langchain_messages = [SystemMessage(content=system_message)]
    for msg in recent:
        if msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        else:
            langchain_messages.append(AIMessage(content=msg["content"]))
    langchain_messages.append(HumanMessage(content=input_text))

    # LLM を呼び出し
    response = st.session_state.llm.invoke(langchain_messages)

    # トークン消費を記録（usage_metadata が取れない場合は概算）
    try:
        usage = response.usage_metadata
        st.session_state.total_tokens += usage["total_tokens"]
    except Exception:
        st.session_state.total_tokens += (len(input_text) + len(response.content)) // 4

    return response.content


def reset_conversation():
    """会話履歴とトークンカウントをリセットする"""
    st.session_state.messages = []
    st.session_state.current_expert = None
    # ※ total_tokens はセッション全体の累計なのでリセットしない


# ==========================================
# UI 画面構築
# ==========================================
st.title("ビジネスプロセス改善アシスタント")
st.write(
    "業務フローの課題を分析し、最新のSaaS・生成AIツールを用いた解決策を提案します。"
)
st.write(
    "STEP1. 左上の「＞ボタン」から設定サイドバーを開く。"
)
st.write(
    "STEP2.専門家モードを選択し、下の入力欄に業務の状況や課題を入力してください。"
)
st.write(
    "💡おすすめの使い方は、最初に業務分析スペシャリストに対して現状の業務プロセスを詳しく説明し、非効率やリスクを洗い出してもらうことです。その後、生成AI・SaaS活用コンサルタントに切り替えて、具体的な改善策を提案してもらう流れが効果的です。"
)


# --- サイドバー ---
with st.sidebar:
    st.header("設定")

    expert_choice = st.radio(
        "専門家モードを選択：",
        ["業務分析スペシャリスト（課題発見・整理）
        業務分析スペシャリスト入力例：営業報告書の作成に毎週3時間かかっています。Excelで管理していますが...",
         "生成AI・SaaS活用コンサルタント（解決策提案）
        業務分析スペシャリストの回答をコピーしたもの "],
    )

    st.divider()

    # リセットボタン（改善案19）
    if st.button("🔄 新しい相談を始める", use_container_width=True):
        reset_conversation()
        st.rerun()

    st.divider()

    # トークン消費量の表示（改善案29）
    st.caption("消費トークン（概算）")
    st.progress(
        min(st.session_state.total_tokens / MAX_TOKENS_PER_SESSION, 1.0)
    )
    st.caption(
        f"{st.session_state.total_tokens:,} / {MAX_TOKENS_PER_SESSION:,} tokens"
    )

# --- 専門家モード切替の通知 ---
if (
    st.session_state.current_expert is not None
    and st.session_state.current_expert != expert_choice
    and st.session_state.messages
):
    st.info("専門家モードを切り替えると会話履歴がリセットされます。")

# --- 会話履歴の表示（改善案2: チャットUI）---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# --- 入力フォーム（改善案3: プレースホルダー）---
user_input = st.chat_input(
    "入力欄。左の設定から入力例を参照しつつ、入力してみましょう。"
)

# --- 送信時の処理 ---
if user_input:
    # ユーザー発言を表示
    with st.chat_message("user"):
        st.write(user_input)

    # LLM に問い合わせ（改善案1: スピナー）
    with st.spinner("専門家が考えています..."):
        try:
            answer = get_llm_response(user_input, expert_choice)
        except Exception as e:
            st.error(
                "申し訳ありません、エラーが発生しました。"
                "しばらく待ってからもう一度お試しください。"
                f"（詳細: {e}）"
            )
            st.stop()

    # AI 回答を表示
    with st.chat_message("assistant"):
        st.write(answer)

    # 履歴に保存
    st.session_state.messages.append({"role": "user",      "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": answer})
