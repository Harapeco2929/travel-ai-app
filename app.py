from flask import Flask, render_template, request, session, redirect, send_from_directory
from database import init_db, save_plan, get_saved_plans, delete_plan, get_plan, save_profile, get_profile
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from openai import OpenAI
import os
import json
from urllib.parse import quote

app = Flask(__name__)
app.secret_key = "travel_ai_secret"
init_db()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
model = tf.keras.models.load_model("travel_model.keras")

class_names = ["city", "culture", "food", "nature", "resort"]

recommendations = {
    "nature": "自然・絶景がお好きな方に合う旅行タイプです。",
    "city": "都市観光がお好きな方に合う旅行タイプです。",
    "culture": "歴史・文化がお好きな方に合う旅行タイプです。",
    "food": "グルメ旅行がお好きな方に合う旅行タイプです。",
    "resort": "リゾート旅行がお好きな方に合う旅行タイプです。"
}

candidate_spots = [
    "上高地", "美瑛", "屋久島", "箱根", "奥多摩", "富士五湖",
    "ツェルマット スイス", "バンフ カナダ", "ミルフォードサウンド ニュージーランド",
    "ソウル 韓国", "釜山 韓国", "台北 台湾", "バンコク タイ",
    "パリ フランス", "ローマ イタリア", "バルセロナ スペイン",
    "京都", "奈良", "鎌倉", "金沢", "浅草",
    "福岡 博多", "大阪 道頓堀", "札幌", "函館", "小樽",
    "沖縄本島", "宮古島", "石垣島", "ハワイ", "バリ島", "済州島"
]

def ask_gpt(prompt):
    response = client.responses.create(
        model="gpt-5-nano",
        input=prompt
    )
    return response.output_text

def extract_json(text):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)

def make_map_url(place):
    return f"https://www.google.com/maps?q={quote(place)}&output=embed"

def create_recommended_spots(prediction):
    prompt = f"""
あなたは旅行先提案AIです。

画像分類結果は「{prediction}」です。
以下の候補から、写真の雰囲気に近そうな旅行先を3つだけ選んでください。
日本でも海外でも構いません。

候補:
{candidate_spots}

必ずJSONだけで返してください。

[
  {{
    "name": "旅行先名",
    "area": "国または地域",
    "description": "30字以内の短い説明"
  }}
]
"""

    try:
        spots = extract_json(ask_gpt(prompt))
    except Exception:
        spots = [
            {"name": "上高地", "area": "長野県", "description": "自然と絶景を楽しめます。"},
            {"name": "バンフ", "area": "カナダ", "description": "山と湖の景色が魅力です。"},
            {"name": "美瑛", "area": "北海道", "description": "写真映えする丘の風景です。"}
        ]

    for spot in spots:
        spot["map"] = make_map_url(spot["name"] + " " + spot["area"])

    return spots

def create_final_schedule(prediction, conditions, chat_history, recommended_spots):
    prompt = f"""
あなたは旅行プランナーです。
以下の情報をすべて反映して、旅行スケジュールを作成してください。

画像分類結果: {prediction}
おすすめ候補地: {recommended_spots}
旅行条件: {conditions}
追加希望: {chat_history}

必ずJSONだけで返してください。説明文やコードブロックは禁止です。

{{
  "destination": "おすすめ目的地",
  "budget": "予算目安",
  "summary": "短い説明",
  "days": [
    {{
      "date": "1日目",
      "items": [
        {{"time": "09:00", "text": "予定内容"}},
        {{"time": "11:00", "text": "予定内容"}},
        {{"time": "13:00", "text": "予定内容"}}
      ]
    }}
  ]
}}

条件:
・各日は3〜5件
・文章は短く
・食事、移動、観光を入れる
・追加希望を必ず反映する
・予算に合う現実的な内容にする
"""

    try:
        return extract_json(ask_gpt(prompt))
    except Exception:
        return {
            "destination": "旅行プラン",
            "budget": conditions.get("budget", "未設定"),
            "summary": "簡易プランを表示します。",
            "days": [
                {
                    "date": "1日目",
                    "items": [
                        {"time": "09:00", "text": "出発"},
                        {"time": "11:00", "text": "観光"},
                        {"time": "13:00", "text": "昼食"},
                        {"time": "17:00", "text": "帰宅または宿泊"}
                    ]
                }
            ]
        }

def short_chat_reply(prediction, conditions, chat_history, user_message):
    recent_history = chat_history[-4:]

    prompt = f"""
あなたは旅行相談AIです。

画像分類結果: {prediction}
旅行条件: {conditions}
直近の会話履歴: {recent_history}
最新のユーザー希望: {user_message}

最重要ルール:
・最新のユーザー希望を最優先する
・ユーザーが言っていない内容を勝手に追加しない
・過去の希望と矛盾する場合は最新の希望を優先する
・温泉と言われていない場合、温泉は提案しない
・カフェと言われた場合、カフェ名を最大3つ提案する
・食べ物を言われた場合、それが食べられる店や地域を最大3つ提案する
・返答は120字以内
・候補は最大3つ
・長くなる場合は質問を1つだけ返す
・おすすめにない旅行先も提案してよい
"""

    try:
        return ask_gpt(prompt)
    except Exception:
        return "候補を追加しました。ほかに重視したいことはありますか？"

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = session.get("prediction")

    if request.method == "POST":

        if "image" in request.files and request.files["image"].filename != "":
            file = request.files["image"]
            filepath = "test_upload.jpg"
            file.save(filepath)

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            predictions = model.predict(img_array)
            predicted_index = np.argmax(predictions)

            prediction = class_names[predicted_index]
            confidence = round(float(predictions[0][predicted_index]) * 100, 2)

            recommended_spots = create_recommended_spots(prediction)

            session["prediction"] = prediction
            session["confidence"] = confidence
            session["recommended_spots"] = recommended_spots
            session["conditions"] = None
            session["final_schedule"] = None
            session["selected_spot"] = None
            session["chat_history"] = [
                {"role": "ai", "message": recommendations[prediction] + " 旅行条件を入力してください。"}
            ]

        elif "spot_name" in request.form:
            spot_name = request.form["spot_name"]

            for spot in session.get("recommended_spots", []):
                if spot["name"] == spot_name:
                    session["selected_spot"] = spot
                    break

        elif "budget" in request.form:
            session["conditions"] = {
                "start_date": request.form["start_date"],
                "departure": request.form["departure"],
                "budget": request.form["budget"],
                "days": request.form["days"],
                "companion": request.form["companion"],
                "transport": request.form["transport"],
                "priority": request.form["priority"]
            }

            session["chat_history"] = [
                {"role": "ai", "message": "条件を保存しました。追加希望があれば入力してください。"}
            ]
            session["final_schedule"] = None

        elif "user_message" in request.form:
            chat_history = session.get("chat_history", [])
            user_message = request.form["user_message"]

            chat_history.append({"role": "user", "message": user_message})

            ai_reply = short_chat_reply(
                session.get("prediction"),
                session.get("conditions"),
                chat_history,
                user_message
            )

            chat_history.append({"role": "ai", "message": ai_reply})
            session["chat_history"] = chat_history

        elif "create_final" in request.form:
            final_schedule = create_final_schedule(
                session.get("prediction"),
                session.get("conditions"),
                session.get("chat_history", []),
                session.get("recommended_spots", [])
            )
            session["final_schedule"] = final_schedule
        elif "save_plan" in request.form:
           final_schedule = session.get("final_schedule")

           if final_schedule:
            title = final_schedule.get("destination", "旅行プラン")
            destination = final_schedule.get("destination", "")
            budget = final_schedule.get("budget", "")
            summary = final_schedule.get("summary", "")

            schedule_text = json.dumps(
               final_schedule,
               ensure_ascii=False
            )

            save_plan(
               title,
               destination,
               budget,
               summary,
               schedule_text
            )
    return render_template(
    "index.html",
    prediction=session.get("prediction"),
    confidence=session.get("confidence"),
    conditions=session.get("conditions"),
    chat_history=session.get("chat_history", []),
    final_schedule=session.get("final_schedule"),
    tourist_spots=session.get("recommended_spots", []),
    selected_spot=session.get("selected_spot"),
    profile=get_profile()
)
@app.route("/saved")
def saved():
    plans = get_saved_plans()
    return render_template(
        "saved.html",
        plans=plans,
        profile=get_profile()
    )

@app.route("/saved/<int:plan_id>")
def saved_detail(plan_id):
    plan = get_plan(plan_id)

    if plan is None:
        return "プランが見つかりませんでした"

    schedule = json.loads(plan["schedule_text"])

    return render_template(
        "saved_detail.html",
        plan=plan,
        schedule=schedule
    )

@app.route("/delete/<int:plan_id>", methods=["POST"])
def delete_saved_plan(plan_id):
    delete_plan(plan_id)
    return redirect("/saved")
@app.route("/profile", methods=["GET", "POST"])
def profile():
    if request.method == "POST":
        nickname = request.form["nickname"]
        age = request.form["age"]
        mbti = request.form["mbti"]
        favorite_travel = request.form["favorite_travel"]

        profile_image = request.files.get("profile_image")
        image_path = None

        if profile_image and profile_image.filename != "":
            filename = profile_image.filename
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            profile_image.save(save_path)
            image_path = filename
            old_profile = get_profile()
        if image_path is None and old_profile:
            image_path = old_profile["image_path"]

        save_profile(
            nickname,
            age,
            mbti,
            favorite_travel,
            image_path
        )

    profile_data = get_profile()

    return render_template(
        "profile.html",
        profile=profile_data
    )
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)