FROM python:3.7
EXPOSE 8501
WORKDIR /app
COPY requirements.txt ./requirements.txt
RUN pip3 install -r requirements.txt
RUN git clone https://github.com/Atlas200dk/sample-imageinpainting-HiFill.git && \
        cd sample-imageinpainting-HiFill && \
        git checkout 1f7f769bd1ea225d4d5c8b094dd261ca9172927b
COPY . .
CMD streamlit run app.py \
    --server.headless true \
    --browser.serverAddress="0.0.0.0" \
    --server.enableCORS false \
    --browser.gatherUsageStats false