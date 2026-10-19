
import os, sqlite3, time, json
from datetime import datetime, timezone
import numpy as np, pandas as pd, requests, streamlit as st, plotly.express as px
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

API="https://pc28.world/api/v1"
DB=os.getenv("CANADA28_DB_PATH","canada28_history.db")
st.set_page_config(page_title="Canada28 多模型分析",page_icon="📊",layout="wide")

def secret(k,default=""):
    try:
        v=st.secrets.get(k,"")
        if v:return str(v).strip()
    except Exception:pass
    return os.getenv(k,default).strip()

def get(path,params=None):
    r=requests.get(f"{API}/{path.lstrip('/')}",params=params,timeout=12)
    r.raise_for_status(); x=r.json()
    if isinstance(x,dict) and x.get("ok") is False: raise RuntimeError(str(x.get("error","API error")))
    return x

def items(p):
    d=p.get("data",p) if isinstance(p,dict) else {}
    if isinstance(d,dict):
        if isinstance(d.get("draw"),dict):return [d["draw"]]
        for k in ("items","draws","results","list"):
            if isinstance(d.get(k),list):return d[k]
        if any(k in d for k in ("nbr","issue","period")):return [d]
    return d if isinstance(d,list) else []

def norm(x):
    issue=x.get("nbr",x.get("issue",x.get("period",x.get("issue_no"))))
    a,b,c=x.get("n1",x.get("digit1")),x.get("n2",x.get("digit2")),x.get("n3",x.get("digit3"))
    nums=x.get("numbers")
    if (a is None or b is None or c is None) and isinstance(nums,(list,tuple)) and len(nums)>=3:a,b,c=nums[:3]
    try:a,b,c=int(a),int(b),int(c)
    except (TypeError,ValueError):return None
    if issue is None or any(n<0 or n>9 for n in (a,b,c)):return None
    s=int(x.get("sum",a+b+c))
    return {"issue":str(issue),"draw_time":str(x.get("time",x.get("draw_time",x.get("date",""))) or ""),
      "n1":a,"n2":b,"n3":c,"sum":s,"big_small":"大" if s>=14 else "小","odd_even":"单" if s%2 else "双",
      "combo":str(x.get("combo",f"{a}{b}{c}")),"source":"pc28.world","synced_at":datetime.now(timezone.utc).isoformat(timespec="seconds")}

def conn():
    c=sqlite3.connect(DB,check_same_thread=False)
    c.execute("""CREATE TABLE IF NOT EXISTS draws(issue TEXT PRIMARY KEY,draw_time TEXT,n1 INTEGER,n2 INTEGER,n3 INTEGER,sum INTEGER,big_small TEXT,odd_even TEXT,combo TEXT,source TEXT,synced_at TEXT)""")
    c.commit();return c

def save(rows):
    if not rows:return
    c=conn()
    c.executemany("""INSERT INTO draws VALUES(:issue,:draw_time,:n1,:n2,:n3,:sum,:big_small,:odd_even,:combo,:source,:synced_at)
      ON CONFLICT(issue) DO UPDATE SET draw_time=excluded.draw_time,n1=excluded.n1,n2=excluded.n2,n3=excluded.n3,sum=excluded.sum,
      big_small=excluded.big_small,odd_even=excluded.odd_even,combo=excluded.combo,source=excluded.source,synced_at=excluded.synced_at""",rows)
    c.commit();c.close()

def sync(limit):
    msgs=[];rows=[]
    try:
        xs=items(get("draws/latest")); r=norm(xs[0]) if xs else None
        if r:rows.append(r);msgs.append("最新开奖期号："+r["issue"])
        else:msgs.append("最新开奖接口返回结构无法识别")
    except Exception as e:msgs.append("最新开奖接口失败："+str(e))
    try:
        xs=items(get("draws",{"limit":limit})); parsed=[norm(x) for x in xs]; rows += [x for x in parsed if x]
        msgs.append(f"历史接口收到 {len(xs)} 条，识别有效记录 {sum(x is not None for x in parsed)} 条")
    except Exception as e:msgs.append("历史接口失败："+str(e))
    save(rows);return msgs

def history():
    c=conn();d=pd.read_sql_query("SELECT * FROM draws",c);c.close()
    if d.empty:return d
    d["_num"]=pd.to_numeric(d.issue,errors="coerce")
    if d["_num"].notna().sum()>=max(2,int(len(d)*.7)):d=d.sort_values(["_num","draw_time"],na_position="last")
    else:d=d.sort_values(["draw_time","issue"],na_position="last")
    return d.drop(columns="_num").reset_index(drop=True)

def feats(d):
    sums=d["sum"].astype(int).to_numpy();out=[]
    for i in range(len(sums)):
        p=sums[max(0,i-20):i]
        if len(p)<5:out.append(None);continue
        f={"last":float(p[-1]),"mean5":float(np.mean(p[-5:])),"std5":float(np.std(p[-5:])),
           "mean10":float(np.mean(p[-10:])),"std10":float(np.std(p[-10:])),
           "mean20":float(np.mean(p[-20:])),"big10":float(np.mean(p[-10:]>=14)),
           "odd10":float(np.mean(p[-10:]%2)),"range10":float(np.ptp(p[-10:]))}
        for lag in range(1,6):f[f"lag{lag}"]=float(p[-lag])
        out.append(f)
    return out

def model_defs():
    return {"逻辑回归":lambda:make_pipeline(StandardScaler(),LogisticRegression(max_iter=800)),
      "随机森林":lambda:RandomForestClassifier(n_estimators=160,max_depth=5,min_samples_leaf=4,random_state=42),
      "梯度提升":lambda:HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,learning_rate=.06,l2_regularization=2,random_state=42)}

def labels(d):
    d=d.copy();d["big_small"]=np.where(d["sum"].astype(int)>=14,"大","小");d["odd_even"]=np.where(d["sum"].astype(int)%2,"单","双");return d

def predict_next(d,target):
    if len(d)<85:return [],"至少建议 85 期历史数据；当前 "+str(len(d))+" 期。"
    d=labels(d).reset_index(drop=True);fs=feats(d);valid=[i for i,x in enumerate(fs) if x is not None]
    dummy=pd.concat([d,pd.DataFrame([{"issue":"NEXT","sum":int(d.iloc[-1]["sum"])}])],ignore_index=True)
    fn=feats(dummy)[-1];names=sorted(fs[valid[0]].keys())
    X=pd.DataFrame([fs[i] for i in valid])[names];xn=pd.DataFrame([fn])[names];y=d.loc[valid,target].astype(str);out=[]
    for name,factory in model_defs().items():
        try:
            m=factory();m.fit(X,y);pred=str(m.predict(xn)[0])
            conf=round(float(max(m.predict_proba(xn)[0])*100),1) if hasattr(m,"predict_proba") else None
            out.append({"模型":name,"分类结果":pred,"模型内置信度(非真实中奖概率)":conf})
        except Exception as e:out.append({"模型":name,"分类结果":"不可用","原因":str(e)})
    return out,None

def backtest(d,target,min_train,max_test):
    if len(d)<min_train+15:return pd.DataFrame(),f"历史数据不足，至少约 {min_train+15} 期；目前 {len(d)} 期。"
    d=labels(d).reset_index(drop=True);fs=feats(d);valid=[i for i,x in enumerate(fs) if x is not None]
    tests=[i for i in valid if i>=min_train][-max_test:];names=sorted(fs[valid[0]].keys());rows=[]
    for i in tests:
        train=[j for j in valid if j<i]
        if len(train)<min_train:continue
        X=pd.DataFrame([fs[j] for j in train])[names];xn=pd.DataFrame([fs[i]])[names];y=d.loc[train,target].astype(str);actual=str(d.loc[i,target])
        majority=y.value_counts().idxmax();rows.append({"期号":d.loc[i,"issue"],"模型":"历史多数类基准","实际":actual,"预测":majority,"命中":int(majority==actual)})
        bit=(i*1103515245+12345)%2
        rand=("大" if bit else "小") if target=="big_small" else ("单" if bit else "双")
        rows.append({"期号":d.loc[i,"issue"],"模型":"固定随机基准","实际":actual,"预测":rand,"命中":int(rand==actual)})
        for name,factory in model_defs().items():
            try:
                m=factory();m.fit(X,y);p=str(m.predict(xn)[0]);rows.append({"期号":d.loc[i,"issue"],"模型":name,"实际":actual,"预测":p,"命中":int(p==actual)})
            except Exception:pass
    return pd.DataFrame(rows),None

def summarize(r):
    if r.empty:return pd.DataFrame()
    s=r.groupby("模型").agg(测试期数=("命中","size"),命中次数=("命中","sum"),命中率=("命中","mean")).reset_index()
    s["命中率(%)"]=(s["命中率"]*100).round(2)
    return s.drop(columns="命中率").sort_values("命中率(%)",ascending=False)

def ai_call(provider,prompt):
    if provider=="ChatGPT":
        key=secret("OPENAI_API_KEY");model=secret("OPENAI_MODEL","gpt-4o-mini")
        if not key:return None,"未配置 OPENAI_API_KEY"
        try:
            r=requests.post("https://api.openai.com/v1/chat/completions",headers={"Authorization":"Bearer "+key},
             json={"model":model,"temperature":0,"messages":[{"role":"system","content":"你是统计分析助手。随机开奖不能被可靠预测；不要编造命中率。"},{"role":"user","content":prompt}]},timeout=30)
            r.raise_for_status();return r.json()["choices"][0]["message"]["content"],None
        except Exception as e:return None,str(e)
    if provider=="Gemini":
        key=secret("GEMINI_API_KEY");model=secret("GEMINI_MODEL","gemini-2.0-flash")
        if not key:return None,"未配置 GEMINI_API_KEY"
        try:
            r=requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",params={"key":key},
             json={"contents":[{"parts":[{"text":"不要声称能可靠预测随机开奖，不要编造成功率。"+prompt}]}]},timeout=30)
            r.raise_for_status();return r.json()["candidates"][0]["content"]["parts"][0]["text"],None
        except Exception as e:return None,str(e)
    if provider=="DeepSeek":
        key=secret("DEEPSEEK_API_KEY");model=secret("DEEPSEEK_MODEL","deepseek-chat")
        if not key:return None,"未配置 DEEPSEEK_API_KEY"
        try:
            r=requests.post("https://api.deepseek.com/chat/completions",headers={"Authorization":"Bearer "+key,"Content-Type":"application/json"},
             json={"model":model,"temperature":0,"messages":[{"role":"system","content":"你是统计分析助手。随机开奖不能被可靠预测；不要编造命中率。"},{"role":"user","content":prompt}]},timeout=45)
            r.raise_for_status();return r.json()["choices"][0]["message"]["content"],None
        except Exception as e:return None,str(e)
    if provider=="Claude":
        key=secret("ANTHROPIC_API_KEY");model=secret("CLAUDE_MODEL","claude-3-5-haiku-latest")
        if not key:return None,"未配置 ANTHROPIC_API_KEY"
        try:
            r=requests.post("https://api.anthropic.com/v1/messages",headers={"x-api-key":key,"anthropic-version":"2023-06-01","content-type":"application/json"},
             json={"model":model,"max_tokens":700,"temperature":0,"system":"随机开奖不能被可靠预测；不要编造命中率。","messages":[{"role":"user","content":prompt}]},timeout=30)
            r.raise_for_status();return "\n".join(x.get("text","") for x in r.json().get("content",[]) if x.get("type")=="text"),None
        except Exception as e:return None,str(e)
    return None,"不支持的 AI 服务："+str(provider)


def ai_recent50_backtest(d, target):
    """Ask each configured AI provider for walk-forward predictions on the latest 50 periods."""
    d = labels(d).reset_index(drop=True)
    start = max(0, len(d) - 50)
    cases = []
    for i in range(start, len(d)):
        prior = d.iloc[max(0, i-20):i]
        if len(prior) < 5:
            continue
        cases.append({"期号": str(d.iloc[i]["issue"]),
                      "此前和值(旧到新)": [int(x) for x in prior["sum"].tolist()]})
    if not cases:
        return {}, "可用于回测的历史数据不足（每个测试期至少需要此前 5 期数据）。"

    target_label = "大小（和值>=14为大，否则小）" if target == "big_small" else "单双（和值为奇数是单，偶数是双）"
    allowed = ["大", "小"] if target == "big_small" else ["单", "双"]
    prompt = (
        "请进行严格的逐期历史回测。对下面每个案例，只能使用该案例给出的此前和值来预测目标分类，"
        "不能使用目标期开奖数据。目标分类规则："+target_label+"。"
        "每个案例必须给出一个预测，取值只能是"+"/".join(allowed)+"。"
        "请只输出合法 JSON，不要 Markdown，不要解释，格式为 "
        '{"predictions":[{"期号":"原期号","预测":"大或小/单或双"}]}。'
        "案例："+json.dumps(cases, ensure_ascii=False, separators=(",", ":"))
    )
    results = {}
    errors = []
    for provider in ["ChatGPT", "Gemini", "Claude", "DeepSeek"]:
        answer, error = ai_call(provider, prompt)
        if error or not answer:
            results[provider] = {"error": error or "没有返回内容"}
            errors.append(provider + "：" + (error or "没有返回内容"))
            continue
        try:
            cleaned = answer.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
            obj = json.loads(cleaned)
            pred_map = {str(x.get("期号")): str(x.get("预测", "")).strip()
                        for x in obj.get("predictions", []) if isinstance(x, dict)}
            rows = []
            for i in range(start, len(d)):
                issue = str(d.iloc[i]["issue"])
                if issue not in {c["期号"] for c in cases}:
                    continue
                pred = pred_map.get(issue, "")
                actual = str(d.iloc[i][target])
                if pred in allowed:
                    rows.append({"期号": issue, "模型": provider, "实际": actual,
                                 "预测": pred, "命中": int(pred == actual)})
            if not rows:
                results[provider] = {"error": "AI 返回内容无法解析为有效预测，请重试。"}
            else:
                frame = pd.DataFrame(rows)
                results[provider] = {"rows": frame, "tested": len(frame),
                                     "hits": int(frame["命中"].sum()),
                                     "accuracy": round(float(frame["命中"].mean() * 100), 2)}
        except Exception as e:
            results[provider] = {"error": "解析 AI 返回失败：" + str(e)}
    return results, ("部分服务失败：" + "；".join(errors)) if errors else None


st.title("📊 Canada28 多模型分析平台")
st.caption("数据源：pc28.world（第三方接口）。统计结果和 AI 输出仅作研究参考，不保证下一期命中或中奖。")
with st.sidebar:
    st.header("开奖同步");limit=st.selectbox("同步历史期数",[100,200,500,1000],index=1)
    if st.button("立即同步开奖、期号和历史",type="primary",use_container_width=True):
        with st.spinner("正在请求 pc28.world…"):st.session_state["syncmsg"]=sync(limit)
    if "syncmsg" not in st.session_state:
        with st.spinner("首次打开时尝试同步…"):st.session_state["syncmsg"]=sync(limit)
    for m in st.session_state["syncmsg"]:st.caption(m)
    st.divider();st.caption("AI 密钥请放在 Streamlit Settings → Secrets，不要上传到 GitHub。")
    st.code('OPENAI_API_KEY = "..."\\nGEMINI_API_KEY = "..."\\nANTHROPIC_API_KEY = "..."\\nDEEPSEEK_API_KEY = "..."\\nDEEPSEEK_MODEL = "deepseek-chat"',language="toml")
d=history()
if d.empty:st.error("暂无历史数据。点击立即同步；若接口失败，请查看左侧错误。");st.stop()
d=labels(d);last=d.iloc[-1]
cols=st.columns(4);cols[0].metric("最新期号",str(last.issue));cols[1].metric("开奖号码",f"{int(last.n1)} + {int(last.n2)} + {int(last.n3)}");cols[2].metric("和值",int(last["sum"]));cols[3].metric("本地历史期数",len(d))
st.caption("开奖时间："+(str(last.draw_time) or "接口未提供"))
tabs=st.tabs(["开奖总览","模型预测","真实历史回测","AI 最近50期准确率","AI 服务","历史记录"])
with tabs[0]:
    st.dataframe(d.tail(30).iloc[::-1][["issue","draw_time","n1","n2","n3","sum","big_small","odd_even"]],use_container_width=True,hide_index=True)
    st.plotly_chart(px.line(d.tail(100),x="issue",y="sum",markers=True,title="最近 100 期和值走势"),use_container_width=True)
    if st.button("读取 pc28.world 第三方预测接口"):
        try:st.session_state["forecast"]=get("forecast");st.session_state.pop("forecast_err",None)
        except Exception as e:st.session_state["forecast_err"]=str(e)
    if st.session_state.get("forecast_err"):st.error(st.session_state["forecast_err"])
    if st.session_state.get("forecast"):st.json(st.session_state["forecast"])
with tabs[1]:
    tl=st.radio("预测目标",["大小","单双","双组组合（预测两组）"],horizontal=True);target="big_small" if tl=="大小" else "odd_even"
    if tl != "双组组合（预测两组）":
        p,err=predict_next(d,target)
        if err:st.warning(err)
        else:st.dataframe(pd.DataFrame(p),use_container_width=True,hide_index=True)
    else:
        st.info("双组模式会让 DeepSeek 从「大单、大双、小单、小双」四种组合中，给出两个不同的下一期参考组合。")
        combo_counts=(d.tail(30).assign(combo_group=lambda x: x["big_small"].astype(str)+x["odd_even"].astype(str))["combo_group"].value_counts())
        st.write("最近 30 期组合频次（仅描述历史）：")
        st.dataframe(pd.DataFrame([{"组合":k,"次数":int(v)} for k,v in combo_counts.items()]),use_container_width=True,hide_index=True)
    st.caption("上方本地模型用于大小/单双分类；置信度不是经过校准的真实中奖概率。")
    st.warning("点击下方按钮才会单独调用 DeepSeek API 并可能产生费用；不会调用其他 AI 服务。")
    if st.button("单独调用 DeepSeek 预测下一期", type="primary", key="deepseek_predict_next"):
        recent = d.tail(30)[["issue", "sum", "big_small", "odd_even"]].to_dict(orient="records")
        if tl == "双组组合（预测两组）":
            prompt = ("请对 Canada28 下一期给出双组组合参考预测。可选组合严格限定为：大单、大双、小单、小双。"
                      "必须选择两个不同组合，第一行明确写‘下一期双组参考预测：组合A、组合B’；再简述最近历史特征。"
                      "大/小按和值>=14为大、否则小；单/双按和值奇偶判断。不要输出第三个组合，不要编造概率或命中率，"
                      "明确说明随机开奖无法可靠预测。最近30期数据："+json.dumps(recent,ensure_ascii=False))
        else:
            target_label = "大小（和值>=14为大，否则小）" if target == "big_small" else "单双（和值为奇数是单，偶数是双）"
            allowed = "大或小" if target == "big_small" else "单或双"
            prompt = ("请基于最近历史记录，对下一期做一个实验性分类估计。目标：" + target_label +
                      "；请给出一个方向（" + allowed + "），并说明参考的统计特征和不确定性。"+
                      "不要编造概率或命中率，明确说明随机开奖无法可靠预测。历史数据："+
                      json.dumps(recent, ensure_ascii=False))
        with st.spinner("正在单独请求 DeepSeek…"):
            answer, deepseek_error = ai_call("DeepSeek", prompt)
        st.session_state["deepseek_prediction_answer"] = answer
        st.session_state["deepseek_prediction_error"] = deepseek_error
    if st.session_state.get("deepseek_prediction_error"):
        st.error("DeepSeek API 调用失败：" + str(st.session_state["deepseek_prediction_error"]))
    elif st.session_state.get("deepseek_prediction_answer"):
        st.markdown("**DeepSeek 下一期参考分析**")
        st.write(st.session_state["deepseek_prediction_answer"])
with tabs[2]:
    st.write("滚动前向回测：每一期只使用该期之前的数据训练。结果是历史表现，不代表未来成功率。")
    tl=st.selectbox("回测目标",["大小命中率","单双命中率"]);target="big_small" if tl.startswith("大小") else "odd_even"
    mt=st.slider("最少训练期数",40,200,80,10);mx=st.slider("最多回测期数",50,500,300,50)
    if st.button("运行真实历史回测",type="primary"):
        with st.spinner("逐期训练回测中…"):r,err=backtest(d,target,mt,mx)
        st.session_state["bt"]=r;st.session_state["bterr"]=err
    r=st.session_state.get("bt",pd.DataFrame());err=st.session_state.get("bterr")
    if err:st.warning(err)
    elif isinstance(r,pd.DataFrame) and not r.empty:
        s=summarize(r);st.dataframe(s,use_container_width=True,hide_index=True)
        st.plotly_chart(px.bar(s,x="模型",y="命中率(%)",text="命中率(%)",title="实际历史回测命中率"),use_container_width=True)
        st.download_button("下载逐期回测 CSV",r.to_csv(index=False).encode("utf-8-sig"),"canada28_backtest.csv","text/csv")
    else:st.info("点击运行回测以计算实际命中率。")
with tabs[3]:
    st.write("调用已配置的 ChatGPT、Gemini、Claude，各自对最近 50 期做逐期回测；每一期只提供该期之前的历史和值，再与实际结果比较。")
    st.caption("每个服务每次回测约调用 1 次 API。结果取决于模型返回的有效预测；随机开奖不可可靠预测，历史命中率不代表未来表现。")
    target_label = st.radio("准确率统计目标", ["大小", "单双"], horizontal=True, key="ai50_target")
    target = "big_small" if target_label == "大小" else "odd_even"
    if st.button("统计 AI 最近 50 期准确率", type="primary", key="run_ai50"):
        with st.spinner("正在调用已配置的 AI 服务并统计最近 50 期…"):
            res, err = ai_recent50_backtest(d, target)
        st.session_state["ai50_results"] = res
        st.session_state["ai50_error"] = err
        st.session_state["ai50_target_label"] = target_label
    if st.session_state.get("ai50_error"):
        st.warning(st.session_state["ai50_error"])
    results = st.session_state.get("ai50_results", {})
    summary = []
    for provider in ["ChatGPT", "Gemini", "Claude", "DeepSeek"]:
        item = results.get(provider)
        if not item:
            continue
        if item.get("error"):
            st.warning(f"{provider}：{item['error']}")
            continue
        summary.append({"AI 服务": provider, "测试期数": item["tested"],
                        "命中次数": item["hits"], "命中率(%)": item["accuracy"]})
    if summary:
        summary_df = pd.DataFrame(summary).sort_values("命中率(%)", ascending=False)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        st.plotly_chart(px.bar(summary_df, x="AI 服务", y="命中率(%)", text="命中率(%)",
                               title=f"AI 最近 50 期{st.session_state.get('ai50_target_label', '大小')}历史命中率"),
                        use_container_width=True)
        detail_frames = [results[n]["rows"] for n in ["ChatGPT", "Gemini", "Claude", "DeepSeek"]
                         if n in results and "rows" in results[n]]
        if detail_frames:
            details = pd.concat(detail_frames, ignore_index=True)
            st.dataframe(details, use_container_width=True, hide_index=True)
            st.download_button("下载 AI 最近50期逐期结果 CSV",
                               details.to_csv(index=False).encode("utf-8-sig"),
                               "ai_recent50_backtest.csv", "text/csv")
    else:
        st.info("点击上方按钮运行统计。需要先在 Streamlit Secrets 配置至少一个 AI 服务密钥。")

with tabs[4]:
    for name,key,model in [("ChatGPT / OpenAI","OPENAI_API_KEY",secret("OPENAI_MODEL","gpt-4o-mini")),("Gemini","GEMINI_API_KEY",secret("GEMINI_MODEL","gemini-2.0-flash")),("Claude","ANTHROPIC_API_KEY",secret("CLAUDE_MODEL","claude-3-5-haiku-latest")),("DeepSeek","DEEPSEEK_API_KEY",secret("DEEPSEEK_MODEL","deepseek-chat"))]:
        st.write(f"**{name}**：{'已配置密钥' if secret(key) else '未配置密钥'} · 模型 `{model}`")
    st.code('OPENAI_API_KEY = "你的密钥"\nOPENAI_MODEL = "gpt-4o-mini"\nGEMINI_API_KEY = "你的密钥"\nGEMINI_MODEL = "gemini-2.0-flash"\nANTHROPIC_API_KEY = "你的密钥"\nCLAUDE_MODEL = "claude-3-5-haiku-latest"\nDEEPSEEK_API_KEY = "你的 DeepSeek API 密钥"\nDEEPSEEK_MODEL = "deepseek-chat"',language="toml")
    st.subheader("DeepSeek 下一期预测")
    ds_target_label = st.radio("预测目标", ["大小", "单双", "双组组合（预测两组）"], horizontal=True, key="deepseek_service_target")
    ds_target = "big_small" if ds_target_label == "大小" else "odd_even"
    if st.button("立即调用 DeepSeek 预测下一期", type="primary", key="deepseek_service_predict"):
        if not secret("DEEPSEEK_API_KEY"):
            st.session_state["deepseek_service_error"] = "未读取到 DEEPSEEK_API_KEY，请检查部署平台的 Secrets。"
            st.session_state["deepseek_service_answer"] = None
        else:
            recent = d.tail(60)[["issue", "sum", "big_small", "odd_even"]].to_dict(orient="records")
            if ds_target_label == "双组组合（预测两组）":
                prompt = ("任务：给出 Canada28 下一期的双组组合参考预测，不要只做历史统计总结。"
                          "可选组合严格限定为大单、大双、小单、小双；必须给出两个不同组合，且第一行明确写‘下一期双组参考预测：组合A、组合B’。"
                          "大/小规则：和值>=14为大，否则小；单/双规则：和值为奇数为单，偶数为双。"
                          "再简述最近数据特征和不确定性。不得声称保证命中，不要编造概率。最近历史记录：" + json.dumps(recent, ensure_ascii=False))
            else:
                target_rule = "和值>=14为大，否则小" if ds_target == "big_small" else "和值为奇数为单，偶数为双"
                prompt = ("任务：给出 Canada28 下一期的单一参考方向，不要只做历史统计总结。目标为" + ds_target_label + "，规则：" + target_rule + "。"
                          "先明确写出‘下一期参考预测：’并只选择一个方向（" + ("大或小" if ds_target == "big_small" else "单或双") + "）；再简述最近数据特征和不确定性。"
                          "不得声称保证命中，不要编造概率。最近历史记录：" + json.dumps(recent, ensure_ascii=False))
            with st.spinner("正在请求 DeepSeek，等待下一期预测结果…"):
                answer, error = ai_call("DeepSeek", prompt)
            st.session_state["deepseek_service_answer"] = answer
            st.session_state["deepseek_service_error"] = error
    if st.session_state.get("deepseek_service_error"):
        st.error("DeepSeek 调用失败：" + str(st.session_state["deepseek_service_error"]))
    elif st.session_state.get("deepseek_service_answer"):
        st.markdown("### DeepSeek 下一期参考预测")
        st.write(st.session_state["deepseek_service_answer"])
    st.divider()
    st.caption("下面是历史描述统计功能，不是下一期预测。该按钮会同时请求所有已配置的 AI 服务，可能分别产生费用。")
    if st.button("调用已配置的 AI 服务（历史统计）"):
        prompt="请总结最近开奖记录的描述性统计、样本限制和不确定性；不要预测确定结果或编造成功率。数据："+d.tail(60)[["issue","sum","big_small","odd_even"]].to_json(orient="records",force_ascii=False)
        st.session_state["airesults"]={n:ai_call(n,prompt) for n in ["ChatGPT","Gemini","Claude","DeepSeek"]}
    for n,(answer,error) in st.session_state.get("airesults",{}).items():
        with st.expander(n,expanded=True):
            if answer:st.write(answer)
            else:st.warning(error)
    st.warning("API 调用可能产生费用。不要把真实密钥放进 app.py 或公开 GitHub。预测仅为参考，随机开奖无法可靠预测。")
with tabs[5]:
    st.dataframe(d.iloc[::-1],use_container_width=True,hide_index=True)
    st.download_button("导出历史 CSV",d.to_csv(index=False).encode("utf-8-sig"),"canada28_history.csv","text/csv")
    st.warning(f"SQLite 文件：{DB}。云端容器重建可能丢失本地文件；长期保存需持久化数据库。")
st.divider();st.caption("本项目仅用于统计研究，不构成投注建议。公平随机开奖无法保证预测。")
