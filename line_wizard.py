import os
import io
from PIL import Image
import google.generativeai as genai
from fastapi import FastAPI, Request, HTTPException
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageMessage

# คลังจำประวัติเดิมของน้า
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

app = FastAPI()

# ===== [ ดึงค่าคีย์ลับอัตโนมัติจากระบบ ] =====
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("google_api_key") or ""
LINE_CHANNEL_ACCESS_TOKEN = os.environ.get("LINE_CHANNEL_ACCESS_TOKEN") or os.environ.get("LINE_CHANNAL_ACCESS_TOKEN") or ""
LINE_CHANNEL_SECRET = os.environ.get("LINE_CHANNEL_SECRET") or os.environ.get("LINE_CHANNAL_SECRET") or ""

# ลงทะเบียนคีย์กับกูเกิลโดยตรงเพื่อใช้ระบบอ่านภาพ
genai.configure(api_key=GOOGLE_API_KEY)

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

@app.post("/callback")
async def callback(request: Request):
    signature = request.headers.get("X-Line-Signature")
    body = await request.body()
    try:
        handler.handle(body.decode("utf-8"), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")
    return "OK"

# ----------------------------------------------------
# 🟢 [ฟังก์ชันข้อความ] สมองกล 4 ด้านขั้นเทพตามสั่ง
# ----------------------------------------------------
@handler.add(MessageEvent, message=TextMessage)
def handle_text_message(event):
    user_message = event.message.text
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)
    
    system_instruction = """
    คุณคือ "Super AI Executive Coach" ที่ปรึกษาธุรกิจและผู้เชี่ยวชาญระดับสูง มีบุคลิกเฉลียวฉลาด สุภาพ เป็นกันเอง และพร้อมช่วยเหลือคู่สนทนาอย่างเต็มที่ โดยคุณมีความเชี่ยวชาญใน 4 ด้านหลักดังนี้:

    1. ที่ปรึกษาและเชี่ยวชาญด้านโมเดลธุรกิจ (Business Model Expert):
       - สามารถวิเคราะห์ ออกแบบ และวิพากษ์ Business Model Canvas, Value Proposition รวมถึงกลยุทธ์การสร้างรายได้ (Revenue Streams) ของธุรกิจทุกรูปแบบได้อย่างเฉียบคม
       - แนะนำกลยุทธ์การแข่งขัน การหาจุดขายที่แตกต่าง (USP) และการปรับเปลี่ยนโมเดลธุรกิจตามเทรนด์โลก

    2. เทคนิคการปิดการขายสินค้าทุกประเภท (Master of Sales Closing):
       - เชี่ยวชาญจิตวิทยาการขาย การโน้มน้าวใจ การตอบข้อโต้แย้ง (Objection Handling) และเทคนิคการปิดการขายแบบเนียนๆ ทั้งแบบ B2B, B2C และออนไลน์
       - สามารถแนะนำสคริปต์การขาย กลยุทธ์การตั้งราคา และจิตวิทยาการสร้างความต้องการให้ลูกค้าอยากซื้อทันที

    3. เทคนิคเขียนโปรเจกต์เพื่อระดมทุนแบบผู้เชี่ยวชาญ (Pitch Deck & Fundraising Specialist):
       - มีความรู้ลึกซึ้งในการเขียนแผนธุรกิจ โครงร่างโปรเจกต์ (Project Proposal) เพื่อขอทุน หรือระดมทุนจาก Venture Capital (VC) และ Angel Investor
       - รู้วิธีการเล่าเรื่อง (Storytelling) การวางโครงสร้าง Pitch Deck ให้น่าดึงดูด และกลยุทธ์การนำเสนอตัวเลขทางการเงินให้เข้าตาผู้ลงทุน

    4. นักจิตวิทยาพัฒนาและดึงศักยภาพของผู้สนทนา (Human Potential & Performance Coach):
       - ใช้หลักจิตวิทยาเชิงบวก (Positive Psychology) และกระบวนการโค้ช (Coaching) เพื่อรับฟัง ตั้งคำถามปลายเปิดเพื่อชวนคิด และสะท้อนมุมมอง
       - ช่วยลดความเครียด สร้างแรงบันดาลใจ ดึงศักยภาพที่ซ่อนอยู่ และช่วยให้คู่สนทนาค้นพบแนวทางแก้ปัญหาหรือเป้าหมายที่ชัดเจนได้ด้วยตัวเอง

    จงตอบคำถามลูกค้าหรือตอบน้าอย่างมืออาชีพ แต่ใช้ภาษาที่เข้าใจง่าย กระชับ นำไปใช้จริงได้ทันที และแฝงความจริงใจในทุกคำตอบ
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])
    
    chain = prompt | model
    demo_ephemeral_chat_history = ChatMessageHistory()
    
    with_message_history = RunnableWithMessageHistory(
        chain,
        lambda session_id: demo_ephemeral_chat_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    
    response = with_message_history.invoke(
        {"input": user_message},
        config={"configurable": {"session_id": event.source.user_id}},
    )
    
    line_bot_api.reply_message(
        event.reply_token,
        TextSendMessage(text=response.content)
    )

# ----------------------------------------------------
# 📸 [ฟังก์ชันรูปภาพ] ดึงภาพสดๆ ในแรม สแกนสลิปโอนเงิน
# ----------------------------------------------------
@handler.add(MessageEvent, message=ImageMessage)
def handle_image_message(event):
    try:
        message_content = line_bot_api.get_message_content(event.message.id)
        image_bytes = io.BytesIO()
        for chunk in message_content.iter_content():
            image_bytes.write(chunk)
        image_bytes.seek(0)
        
        img = Image.open(image_bytes)
        
        analysis_prompt = """
        คุณคือระบบตรวจสอบสลิปโอนเงินอัจฉริยะของร้าน 
        จงตรวจสอบรูปภาพนี้ว่าเป็นสลิปโอนเงินของธนาคารไทย หรือสลิป TrueMoney ใช่หรือไม่?
        ถ้าใช่ ให้สรุปข้อมูลสั้นๆ เป็นกันเองและชัดเจนตอบกลับลูกค้าดังนี้:
        - ยอดเงินโอน (บาท)
        - วันที่และเวลาที่โอนสำเร็จ
        - ชื่อผู้โอนเงิน
        
        ตัวอย่างการตอบ: "ได้รับยอดโอนเงินเรียบร้อยแล้วครับน้า! ยอดเงิน 500 บาท จากคุณ ธวัชชัย ระบบลงบันทึกให้เรียบร้อยครับ"
        แต่ถ้าไม่ใช่รูปสลิป ให้ตอบว่า "ขออภัยครับน้า ภาพนี้ดูเหมือนไม่ใช่สลิปโอนเงินที่ถูกต้อง กรุณาลองส่งใหม่อีกครั้งนะครับ"
        """
        
        vision_model = genai.GenerativeModel('gemini-2.5-flash')
        response = vision_model.generate_content([analysis_prompt, img])
        
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=response.text)
        )
        
    except Exception as e:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=f"บอทสแกนภาพสะดุดไปนิดนึงครับน้า (เนื่องจาก: {str(e)}) ลองส่งรูปอีกทีนะครับ")
        )

# บรรทัดสตาร์ตรันระบบสำหรับเครื่องข่าย FastAPI ท้ายไฟล์อย่างถูกต้อง
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("line_wizard:app", host="0.0.0.0", port=10000, reload=True)