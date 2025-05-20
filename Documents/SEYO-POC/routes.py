
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
import json
from typing import List, Union, Optional
from pydantic import BaseModel
from QA_bot import ChatService

router = APIRouter()

class ChatHistoryModel(BaseModel):
    isUser: bool
    content: Union[str, dict]

@router.post("/qa")
async def qa_bot(
    user_request: str = Form(...),
    data: Optional[str] = Form(None),
    chat_history: Optional[str] = Form(None),
    file_attachment: Optional[UploadFile] = File(None)
):
    try:
        # Handle data parameter more defensively
        if data is not None and data.strip():
            try:
                data_dict = json.loads(data)
            except json.JSONDecodeError:
                data_dict = {}
        else:
            data_dict = {}
            
        # Handle chat history parameter more defensively
        if chat_history is not None and chat_history.strip():
            try:
                chat_history_data = json.loads(chat_history)
                chat_history_parsed = [
                    ChatHistoryModel(isUser=item.get("isUser", False),
                                   content=item.get("content", ""))
                    for item in chat_history_data
                ]
            except json.JSONDecodeError:
                chat_history_parsed = []
        else:
            chat_history_parsed = []
            
        # Read PDF bytes
        pdf_bytes = await file_attachment.read() if file_attachment else None
        
        # Get answer
        answer = ChatService.get_chatresponse(
            pdf_bytes=pdf_bytes,
            message=user_request,
            chat_history=chat_history_parsed,
            data=data_dict
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))