# APP_NAME=banknote-api
# PORT=5000

# .PHONY: run docker-build docker-run docker-stop clean

# run:
# 	python flask_app.py

# docker-build:
# 	docker build -t $(APP_NAME) .

# docker-run:
# 	docker run -p $(PORT):5000 --name $(APP_NAME) $(APP_NAME)

# docker-stop:
# 	docker stop $(APP_NAME) || true
# 	docker rm $(APP_NAME) || true

# clean:
# 	docker rmi $(APP_NAME) || true
