import os
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

app = FastAPI()

# ======================== [ ดึงค่าคีย์ลับอัตโนมัติจากระบบ ] ========================
import os

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("google_api_key") or ""

LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN") or os.environ.get("LINE_CHANNAL_ACCESS_TOKEN") or os.environ.get("line_channel_access_token") or ""
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET") or os.environ.get("LINE_CHANNAL_SECRET") or os.environ.get("line_channel_secret") or os.environ.get("line_channal_secret") or ""

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# ====================================================================================
# บังคับอัปเดตระบบป้องกัน Langchain เออร์เรอร์
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# ====================================================================================
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# ====================================================================================
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY if GOOGLE_API_KEY else ""
# ====================================================================================

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

    from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage
import tempfile

# 1. เพิ่มฟังก์ชันสำหรับจัดการเมื่อมี "รูปภาพ" ส่งเข้ามาใน LINE
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    # ดึง ID ของข้อความรูปภาพ
    message_id = event.message.id
    
    # ดาวน์โหลดไฟล์รูปภาพจาก LINE เก็บไว้ชั่วคราว
    message_content = line_bot_api.get_message_content(message_id)
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tf:
        for chunk in message_content.iter_content():
            tf.write(chunk)
        temp_file_path = tf.name

    try:
        # 2. เรียกใช้ Gemini Vision เพื่ออ่านรูปสลิป
        # (หมายเหตุ: คำสั่งด้านล่างนี้เป็นการใช้สำหรับกูเกิลเจเนอเรทีฟเอไอ)
        import google.generativeai as genai
        from PIL import Image
        
        # เปิดรูปภาพที่เซฟไว้
        img = Image.open(temp_file_path)
        
        # ตั้งคำสั่ง (Prompt) ให้ Gemini แกะข้อมูลสลิปอย่างแม่นยำ
        prompt = """
        คุณคือระบบตรวจสอบสลิปโอนเงินอัจฉริยะ 
        จงตรวจสอบรูปภาพนี้ว่าเป็นสลิปโอนเงินของธนาคารไทยใช่หรือไม่?
        ถ้าใช่ ให้ดึงข้อมูลต่อไปนี้ออกมา:
        1. ชื่อธนาคาร
        2. วันที่และเวลาที่โอน
        3. จำนวนเงิน (บาท)
        4. ชื่อผู้โอน และ ชื่อผู้รับเงิน
        
        แล้วสรุปตอบกลับลูกค้าสั้นๆ เป็นกันเอง เช่น "ได้รับสลิปเรียบร้อยครับ ยอดโอน XX บาท จากคุณ XX ไปยัง XX วันที่ XX"
        แต่ถ้าภาพนี้ไม่ใช่สลิปโอนเงิน ให้ตอบสุภาพว่า "ขออภัยครับ ภาพนี้ไม่ใช่สลิปโอนเงิน กรุณาส่งรูปสลิปใหม่อีกครั้งนะครับ"
        """
        
        # สั่งให้ Gemini ประมวลผลภาพ
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content([prompt, img])
        
        # 3. ส่งคำตอบกลับไปหาลูกค้าใน LINE
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=response.text)
        )
        
    except Exception as e:
        print(f"Error processing image: {e}")
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text="ขออภัยครับ ระบบเกิดข้อผิดพลาดในการอ่านรูปภาพ ลองใหม่อีกครั้งนะครับ")
        )