# image-gen-agent

a2hmarket 沙箱 agent。Prompt → OpenAI `gpt-image-2-2026-04-21` → PNG。免费，不开单。

## 协议

- `POST /chat` — 接收 `req.text` 作为 prompt，生成 1 张 PNG 上传到 `chatfile/image`，通过 `ui("show_file", ...)` 回给 chatbox。
- `GET /health` — worker pool 健康探针。

单张图生成 5-20s；worker `concurrency=3`。

## 凭证

OpenAI API key 存在 SSM：`/a2h/agents/image-gen/openai-api-key`（SecureString）。

Agent 启动时 boto3 拉一次，缓存在进程里。本地调试可以用 env `OPENAI_API_KEY` 直接覆盖。

## 本地冒烟

```bash
# 本机需要的 env（平台会在容器里注入 bypass key + seller id）
export OPENAI_API_KEY=sk-proj-...
export A2H_TOKEN=dummy
export A2H_GATEWAY_BYPASS=<bypass-key>
export A2H_SELLER_ID=<seller-cognito-sub>
export A2H_PLATFORM_BASE=http://findu-alb-476446960.us-east-1.elb.amazonaws.com

pip install -e ../../Workspace/aws_codebase/side_project/diy_shop/kit-v2/agent-sdk
pip install -r requirements.txt

uvicorn server:app --port 8080

# 另一个终端
curl -N -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "smoke-1",
    "shop_id": 0,
    "buyer": {"id": "u-test"},
    "history": [],
    "message": {"text": "a baby otter with a stethoscope, childrens book drawing"}
  }'
```

## 部署到沙箱（由平台侧执行）

1. push 到 `xemaya/image-gen-agent`（public 或给 CodeBuild 机器人 read 权限）
2. 触发 CodeBuild：
   ```bash
   aws codebuild start-build --project-name a2h-agent-builder \
     --environment-variables-override \
       name=PACKAGE_ID,value=<new-pkg> \
       name=SHOP_ID,value=<bound-shop> \
       name=GIT_URL,value=https://github.com/xemaya/image-gen-agent.git \
       name=GIT_REF,value=main
   ```
3. 激活 worker pool：`POST /findu-diy-shop/api/v1/shops/{SHOP}/worker-pool/activate`
