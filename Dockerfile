# Cocoon Authority Glass Box - HF Space (Docker). The Authority serves the UI +
# the engine on HF compute, and renders the real Glass dashboard HEADLESS
# (SDL dummy driver) in-process, streamed to the page at /glass.mjpg.
# No Xvfb, no VNC, no screen-grab - pure app, HF-policy-safe.
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    SDL_VIDEODRIVER=dummy \
    DISPLAY=:0

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

COPY --chown=user:user requirements.txt ./
RUN pip install --no-cache-dir --user -r requirements.txt \
    && pip install --no-cache-dir --user "pygame>=2.5.0" "Pillow>=10.0.0"

COPY --chown=user:user . .

EXPOSE 7860
CMD ["python", "mira_kite_authority.py", "--cocoon", "cocoon_cognition_agency.py", "--host", "0.0.0.0", "--port", "7860"]
