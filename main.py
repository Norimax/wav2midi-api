from fastapi import FastAPI, File, UploadFile
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH
import shutil
from fastapi.responses import FileResponse

app = FastAPI()

@app.post("/convert")
async def convert_audio(file: UploadFile = File(...)):
    input_audio = f"/tmp/{file.filename}"
    output_dir = "/tmp/"
    with open(input_audio, "wb") as f:
        shutil.copyfileobj(file.file, f)

    predict_and_save(
        [input_audio],
        output_dir,
        save_midi=True,
        sonify_midi=False,
        save_notes=False,
        save_model_outputs=False,
        model_or_model_path=ICASSP_2022_MODEL_PATH
    )

    output_midi = input_audio.replace('.wav', '_basic_pitch.mid')
    return FileResponse(output_midi, media_type="audio/midi", filename="output.mid")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
