import argparse
import uvicorn
import gradio as gr
from fastapi.responses import RedirectResponse

from app.main import app as fastapi_app
from frontend.app import create_gradio_app


def main():
    parser = argparse.ArgumentParser(description="Customer Churn Prediction & Retention Engine")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port for server (default: 8000)")
    parser.add_argument("--api-only", action="store_true", help="Launch FastAPI backend only")
    parser.add_argument("--ui-only", action="store_true", help="Launch Gradio UI only")
    args = parser.parse_args()

    if args.ui_only:
        print(f"Starting standalone Gradio UI on http://{args.host}:7860 ...")
        ui = create_gradio_app()
        ui.launch(server_name=args.host, server_port=7860, show_api=False)
    elif args.api_only:
        print(f"Starting FastAPI backend only on http://{args.host}:{args.port} ...")
        print(f"Interactive Swagger Docs: http://{args.host}:{args.port}/docs")
        uvicorn.run(fastapi_app, host=args.host, port=args.port)
    else:
        print("Mounting Gradio UI onto FastAPI backend...")
        ui = create_gradio_app()
        mounted_app = gr.mount_gradio_app(fastapi_app, ui, path="/ui")

        # Redirect root / to /ui
        @mounted_app.get("/", include_in_schema=False)
        async def redirect_root():
            return RedirectResponse(url="/ui")

        print("=" * 65)
        print("🚀 CUSTOMER CHURN SEQUENTIAL & RETENTION PLATFORM")
        print("=" * 65)
        print(f"🌐 Full-Stack Application: http://{args.host}:{args.port}/")
        print(f"🖥️  Gradio Interactive UI: http://{args.host}:{args.port}/ui")
        print(f"📚 FastAPI Swagger Docs : http://{args.host}:{args.port}/docs")
        print(f"🔍 ReDoc Documentation  : http://{args.host}:{args.port}/redoc")
        print("=" * 65)

        uvicorn.run(mounted_app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
