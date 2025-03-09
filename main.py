from fastapi import FastAPI, File, UploadFile, HTTPException
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH
import shutil
import uvicorn
import os

app = FastAPI()

@app.post("/convert")
async def convert_audio(file: UploadFile = File(...)):
    input_audio = f"/tmp/{file.filename}"
    output_midi = input_audio.replace('.wav', '_basic_pitch.mid')
    
    try:
        # 🔥 既存のMIDIファイルを削除
        if os.path.exists(output_midi):
            os.remove(output_midi)

        # WAVファイルを保存
        with open(input_audio, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Basic Pitchの処理
        try:
            predict_and_save(
                [input_audio],
                "/tmp/",
                save_midi=True,
                sonify_midi=False,
                save_notes=False,
                save_model_outputs=False,
                model_or_model_path=ICASSP_2022_MODEL_PATH
            )
        except Exception as e:
            print(f"🚨 Basic Pitch エラー: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Basic Pitch Error: {str(e)}")

        return {"midi_path": output_midi}
    
    except Exception as e:
        print(f"🚨 サーバーエラー: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        timeout_keep_alive=300  # 🔥 タイムアウトを5分に延長
    )
