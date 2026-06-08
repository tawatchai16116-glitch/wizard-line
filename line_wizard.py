import os
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

app = FastAPI()

# ==================== [ ตั้งค่า KEY ต่างๆ ตรงนี้ ] ====================
# 1. ใส่ Google Gemini API Key เดิมของคุณ (ตัวที่ดึงมาจาก Google AI Studio)
os.environ["GOOGLE_API_KEY"] = ""

# 2. ใส่คีย์ลับของ LINE ที่คุณเปิดเจอในหน้าเว็บ LINE Developers
LINE_CHANNEL_ACCESS_TOKEN = "".strip()
LINE_CHANNEL_SECRET = "".strip()
# ==================================================================

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# 3. โครงสร้างสมองธุรกิจและจิตวิทยาตัวเดิมของคุณ
system_instruction = """
คุณคือ "wizard" AI อัจฉริยะที่เป็นส่วนผสมของ นักกลยุทธ์ธุรกิจชั้นยอด (Business Strategist) และ นักจิตวิทยาพฤติกรรมมนุษย์ (Behavioral Psychologist)

แนวทางในการวิเคราะห์และตอบคำถามของคุณ:
1. วิเคราะห์จุดอ่อนธุรกิจอย่างตรงไปตรงมา: มองหา Pain points, Bottlenecks หรือสิ่งที่ไม่สมเหตุสมผลในโมเดลธุรกิจ
2. ใช้จิตวิทยาขับเคลื่อน: นำทฤษฎีจิตวิทยามาประยุกต์ใช้เสมอ เช่น Loss Aversion, Social Proof หรือ Hook Model
3. ระบบเรียนรู้และประเมินตัวเอง (Self-Reflection): ในทุกๆ ท้ายคำตอบ จงสร้างหัวข้อ "[บันทึกการเรียนรู้ของ wizard]" เพื่อสรุปบทเรียนใหม่ๆ

ตอบด้วยภาษาไทยที่เฉียบคม มีพลัง และโฟกัสที่การเติบโต (Growth Mindset)
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_instruction),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)
chain = prompt | llm

chat_store = {}
def get_session_history(session_id: str):
    if session_id not in chat_store:
        chat_store[session_id] = ChatMessageHistory()
    return chat_store[session_id]

anima_agent = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# เส้นทางสำหรับให้ LINE ยิงสัญญาณ (Webhook) เข้ามา
@app.post("/webhook")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = await request.body()
    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return "OK"

# ฟังก์ชันจัดการเมื่อมีคนส่งข้อความตัวอักษรเข้ามาใน LINE
@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_text = event.message.text
    # ใช้ User ID ของไลน์แต่ละคนเป็น Session ID เพื่อให้ AI จำแยกคนได้
    user_id = event.source.user_id 
    
    # ส่งข้อความไปถาม Gemini wizard
    response = anima_agent.invoke(
        {"input": user_text},
        config={"configurable": {"session_id": user_id}}
    )
    
    # ส่งคำตอบกลับไปหาผู้ใช้บนแอป LINE
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=response.content)
    )

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("line_wizard:app", host="0.0.0.0", port=port)