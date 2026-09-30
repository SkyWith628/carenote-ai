# =============================================================================
# 케어노트AI — Web (Next.js) 이미지
#
# [What] deps → builder → runner 3단계. runner 는 standalone 산출물만 담는다.
# [Why]  node_modules 전체가 아닌 Next standalone 번들만 실어 이미지 최소화,
#        non-root(nextjs) 실행으로 피해 최소화.
# [How]  Next.js 공식 Docker 권장 구성. next.config 에 output:'standalone' 필요.
#        채워야 할 곳은 TODO.
# =============================================================================

# ---- 1) deps: 의존성 설치 (락파일 기준 캐시) ---------------------------------
FROM node:20-slim AS deps
WORKDIR /app
# TODO: package.json 을 채운 뒤 락파일과 함께 복사 (npm/pnpm/yarn 중 택1)
COPY package.json ./
# COPY package-lock.json ./     # npm
RUN echo "TODO: npm ci  (락파일 생기면 활성화)" && mkdir -p node_modules

# ---- 2) builder: 빌드 --------------------------------------------------------
FROM node:20-slim AS builder
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY . .
# TODO: next.config.* 에 `output: 'standalone'` 설정 후 활성화
RUN echo "TODO: npm run build"

# ---- 3) runner: 실행 (standalone 산출물만) -----------------------------------
FROM node:20-slim AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000

# non-root 유저
RUN groupadd -r nodejs && useradd -r -g nodejs -u 10001 nextjs

# standalone 번들 + 정적 자산 (빌드 활성화하면 아래 3줄 주석 해제)
# COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
# COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
# COPY --from=builder --chown=nextjs:nodejs /app/public ./public

USER nextjs
EXPOSE 3000

HEALTHCHECK --interval=15s --timeout=5s --retries=5 --start-period=20s \
    CMD node -e "require('http').get('http://localhost:3000',r=>process.exit(r.statusCode<500?0:1)).on('error',()=>process.exit(1))"

# standalone 진입점 (빌드 후 활성화)
# CMD ["node", "server.js"]
CMD ["sh", "-c", "echo 'TODO: web 빌드 채운 뒤 CMD를 node server.js 로 교체' && sleep infinity"]
