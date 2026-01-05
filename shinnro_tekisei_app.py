import streamlit as st
import pandas as pd
import numpy as np
import os
import random 
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go

# --- 1. 日本語変換用の完全マップ ---
JP_MAP = {
    "logical_thinking": "論理力",
    "logical thinking": "論理力",
    "communication": "対人力",
    "creativity": "創造力",
    "meticulousness": "正確性",
    "stress_tolerance": "耐性",
    "stress tolerance": "耐性"
}

def to_jp(eng_name):
    """英語の項目名を日本語に変換する"""
    key = str(eng_name).lower().strip().replace(' ', '_')
    return JP_MAP.get(key, eng_name)

# --- 2. レーダーチャート作成関数 ---
def plot_radar_chart(user_scores, job_scores, labels, job_name):
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=user_scores,
        theta=labels,
        fill='toself',
        name='あなたの特性',
        line_color='#1f77b4'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=job_scores,
        theta=labels,
        fill='toself',
        name=f'{job_name}の基準',
        line_color='#7f7f7f',
        opacity=0.6
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
        showlegend=True,
        title=dict(text=f"特性バランス比較: あなた vs {job_name}", font=dict(size=18))
    )
    return fig

# --- 3. AI深層分析・解説生成ロジック ---
def generate_ai_analysis(best_match_job, user_scores_id, p_df, axis_ids):
    prof = p_df[p_df['ProfileName'] == best_match_job].iloc[0]
    
    details = []
    for aid in axis_ids:
        if aid in p_df.columns:
            u_val = user_scores_id[aid]
            p_val = float(prof[aid])
            details.append({
                "id": aid,
                "name": to_jp(aid),
                "diff": abs(u_val - p_val),
                "u_val": u_val,
                "p_val": p_val
            })
    
    sorted_details = sorted(details, key=lambda x: x['diff'])
    top = sorted_details[0]   
    second = sorted_details[1]

    trait_insights = {
        "論理力": ["複雑な課題を構造化し、最短ルートで解決する力", "データに基づいた客観的な判断軸"],
        "対人力": ["組織の潤滑油となり、チームの生産性を最大化する調整力", "相手のニーズを先読みする共感力"],
        "創造力": ["既存の枠組みに捉われず、新しい価値を生み出す発想力", "常識を疑い、変化を楽しむ柔軟性"],
        "正確性": ["細部にまでこだわり、組織の信頼を支える完遂力", "ミスを未然に防ぐプロフェッショナルな規律"],
        "耐性": ["プレッシャーを成長の糧に変え、粘り強く成果を出す精神性", "変化の激しい環境でも自分を見失わない安定感"]
    }

    def get_eval_comment(trait_data):
        if trait_data['diff'] < 0.3:
            return "驚くほど理想的な水準に達しており、即戦力としての資質を感じさせます。"
        elif trait_data['u_val'] > trait_data['p_val']:
            return "求められる基準を上回るポテンシャルがあり、リーダーとして周囲を牽引できるレベルです。"
        else:
            return "バランスが非常に良く、安定して能力を発揮できる準備が整っています。"

    if any(k in best_match_job for k in ["コンサル", "企画", "研究"]):
        role_type = "戦略的アドバイザー"
    elif any(k in best_match_job for k in ["制作", "開発", "デザイナー"]):
        role_type = "クリエイティブ・リーダー"
    elif any(k in best_match_job for k in ["事務", "管理", "公務員"]):
        role_type = "組織の安定を支える守護神"
    else:
        role_type = "現場を牽引するプロフェッショナル"

    intro = [
        f"AIの多次元解析により、あなたの本質は{best_match_job}の中核的な要素と深く共鳴していることが判明しました。",
        f"全職業データとのシミュレーションにおいて、あなたの特性は{best_match_job}として活躍する上で最も効率的な形を示しています。"
    ]

    analysis_text = f"""
    【AI分析レポート】
    {random.choice(intro)}

    **■ 核心的な適合要素：{top['name']}**
    あなたの持つ「{top['name']}」は、{random.choice(trait_insights.get(top['name'], ['優れた資質']))}と言い換えることができます。
    {get_eval_comment(top)}

    **■ 相乗効果を生むサブ特性：{second['name']}**
    次に注目すべきは「{second['name']}」の高さです。これが{top['name']}と組み合わさることで、単なる適性を超えて、**「あなたにしかできない{best_match_job}のスタイル」**を確立できるはずです。

    **■ キャリアの展望**
    AIは、あなたが現場で**「{role_type}」**として周囲から頼りにされる姿を予測しています。
    特に「{top['name']}」を意識して活動することで、成長スピードは飛躍的に高まるでしょう。
    """
    return analysis_text

# --- 4. データの読み込み ---
@st.cache_data
def load_data():
    path_q = '/content/drive/MyDrive/Colab Notebooks/職業/questions_axis.csv'
    path_p = '/content/drive/MyDrive/Colab Notebooks/職業/profiles.csv'
    if not os.path.exists(path_q): return None, None
    
    def read_df(p):
        try: return pd.read_csv(p, encoding='utf-8')
        except: return pd.read_csv(p, encoding='shift_jis')
        
    q_df, p_df = read_df(path_q), read_df(path_p)
    q_df.columns = q_df.columns.str.strip()
    p_df.columns = p_df.columns.str.strip()
    
    temp_cols = list(q_df.columns)
    q_df = q_df.rename(columns={temp_cols[0]: 'AxisID', temp_cols[1]: 'AxisName', temp_cols[2] if len(temp_cols) > 2 else temp_cols[-1]: 'Question'})
    p_df = p_df.rename(columns={'job_name': 'ProfileName'})
    
    return q_df, p_df

# --- 5. メイン画面 ---
def main():
    st.set_page_config(page_title="進路適性診断 AI", layout="centered")
    st.title("🎓 進路適性診断システム (scikit-learn版)")
    
    q_df, p_df = load_data()
    if q_df is None:
        st.error("CSVファイルが読み込めませんでした。Driveのパスを確認してください。")
        return

    with st.form(key='aptitude_form'):
        st.header("📝 診断質問")
        for i, row in q_df.iterrows():
            axis_label = to_jp(row['AxisID'])
            st.slider(f"**【{axis_label}】**\n{row['Question']}", 1, 5, 3, key=f"q_{i}")
        submitted = st.form_submit_button("AI診断を実行")

    if submitted:
        axis_ids = q_df['AxisID'].unique()
        user_scores_id = {aid: np.mean([st.session_state[f"q_{idx}"] for idx in q_df[q_df['AxisID'] == aid].index]) for aid in axis_ids}
        
        common_axes = [aid for aid in axis_ids if aid in p_df.columns]
        user_vector = np.array([[user_scores_id[aid] for aid in common_axes]])

        match_results = []
        for _, prof in p_df.iterrows():
            job_vector = np.array([[float(prof[aid]) for aid in common_axes]])
            similarity = cosine_similarity(user_vector, job_vector)[0][0]
            match_results.append({'職業': prof['ProfileName'], 'マッチ度': similarity * 100, 'raw_data': prof})

        if match_results:
            res_df = pd.DataFrame(match_results).sort_values('マッチ度', ascending=False)
            best_match = res_df.iloc[0]
            worst_match = res_df.iloc[-1]

            st.header("📊 診断結果")
            st.success(f"### ✨ 最も適性がある職業: {best_match['職業']}")
            
            # --- 修正箇所：適合率 → マッチ度 ---
            col1, col2 = st.columns([1, 2])
            col1.metric("マッチ度", f"{best_match['マッチ度']:.1f}%")
            col2.progress(min(best_match['マッチ度'] / 100, 1.0))
            st.balloons()

            st.divider()
            st.subheader("🕸️ 特性バランスの視覚化")
            labels_jp = [to_jp(aid) for aid in common_axes]
            user_vals = [user_scores_id[aid] for aid in common_axes]
            job_vals = [float(best_match['raw_data'][aid]) for aid in common_axes]
            fig = plot_radar_chart(user_vals, job_vals, labels_jp, best_match['職業'])
            st.plotly_chart(fig, use_container_width=True)

            st.divider()
            st.subheader("🤖 AI分析アドバイス")
            with st.spinner('データを詳細に解析中...'):
                advice = generate_ai_analysis(best_match['職業'], user_scores_id, p_df, axis_ids)
                st.info(advice)

            st.divider()
            with st.expander("⚠️ 参考：現在の特性と「最も異なる」職業を表示"):
                st.write(f"あなたの今の傾向とは正反対のスタイルを持つ職業は **「{worst_match['職業']}」** です。")
                # --- 修正箇所：適合率 → マッチ度 ---
                st.write(f"（マッチ度: {worst_match['マッチ度']:.1f}%）")

            st.divider()
            # --- 修正箇所：適合率 → マッチ度 ---
            st.subheader("📈 職業別マッチ度ランキング（全件）")
            ranking_display = res_df[['職業', 'マッチ度']].copy()
            st.dataframe(ranking_display.style.format({'マッチ度': '{:.1f}%'}), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()