# 케어노트AI — 로컬 개발 단축 명령
# compose 파일이 infra/ 에 있어 -f 경로를 매번 붙이는 대신 여기에 묶는다.
COMPOSE := docker compose -f infra/docker-compose.yml

.PHONY: help up up-app down logs ps psql redis-cli clean

help: ## 명령 목록
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-10s\033[0m %s\n",$$1,$$2}'

up: ## 인프라만 기동 (postgres+redis) — Phase1-①
	$(COMPOSE) up -d

up-app: ## api·web 포함 전체 기동 (코드 채운 뒤)
	$(COMPOSE) --profile app up -d --build

down: ## 전체 종료 (볼륨 유지)
	$(COMPOSE) down

logs: ## 로그 팔로우
	$(COMPOSE) logs -f

ps: ## 컨테이너 상태
	$(COMPOSE) ps

psql: ## postgres 접속
	$(COMPOSE) exec postgres psql -U carenote -d carenote

redis-cli: ## redis 접속
	$(COMPOSE) exec redis redis-cli

clean: ## 전체 종료 + 볼륨 삭제 (데이터 초기화, 주의)
	$(COMPOSE) down -v
