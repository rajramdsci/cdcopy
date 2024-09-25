
import os,re
import json
import pandas as pd
import templates
from flask import Flask, request
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from gen_ai_hub.proxy.langchain.init_models import init_llm
from gen_ai_hub.proxy.native.openai import embeddings

app = Flask(__name__)

def get_embedding(input, model="text-embedding-ada-002"):
    response = embeddings.create(model_name=model,input=input)
    return str(response.data[0].embedding)

def formatted_json_response(input_text, model="gpt-35-turbo"):
    try:
        good_example = """[{"scenario":"","calculation":"","cost":""},{"scenario":"","calculation":"","cost":""}]"""
        fail_example = """[{"scenario":"NA","calculation":"NA","cost":"NA"}]"""
        prompt = PromptTemplate(template=templates.format_prompt, input_variables=['s_example', 'f_example', 'input_text'])
        llm = init_llm(model, max_tokens=1800)
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        nresponse = llm_chain.invoke({'s_example': good_example, 'f_example': fail_example, 'input_text': input_text})
        fresponse = json.loads(nresponse['text'])
        return fresponse
    except Exception as err:
        print("Something Went Wrong While Formating!!")
        return [{"scenario": "NA", "calculation": "NA", "cost": "NA"}]


def extract_demand_event_time(user_query):
    class UserIntent(BaseModel):
        demand: str = Field(description="Energy Demand in kwh or 'NA' if not found")
        event: str = Field(description="Cause or reason for Energy demand or 'NA' if not found")
        when: str = Field(description="Time - when energy demand is required or 'NA' if not found")

    llm = init_llm('gpt-35-turbo', max_tokens=200)
    model_with_structure = llm.with_structured_output(UserIntent)
    userintent = model_with_structure.invoke(user_query)
    return userintent.dict()


@app.route("/v1/query_model/",methods=['POST'])
def query_model():
    try:
        requestbody = request.get_json()
        userquery = requestbody['userquery']
        user_info = extract_demand_event_time(userquery)
        embedding = get_embedding(userquery,model="text-embedding-ada-002")
        return {"status":"Success","userquery":userquery, "triggerkeyword": user_info['event'],"demand":user_info['demand'],"timeframe": user_info['when'],"queryembedding" : embedding }
    except Exception as err:
        print(err)
        return {"status":"Success", "userquery": 'NA', "triggerkeyword": "NA","demand": "NA", "timeframe": "NA", "queryembedding": "[]"}


@app.route("/v1/optimized_cost_scenario/",methods=['POST'])
def optimized_cost_scenario():
    try:
        requestbody = request.get_json()
        userquery = requestbody['userquery']
        demand = int(requestbody['totalDemand'])
        triggerkeyword = requestbody['triggerKeyword']
        timeframe = requestbody['timeframe']
        model = requestbody['model']
        df = pd.DataFrame(requestbody['master_optimisations'])
        df = df[['Optimisation_Scenario', 'DSR_P', 'Renewables_P', 'Battery_P', 'Spot_P']]
        df.columns = df.columns.str.replace('_P', ' % ')
        x = df.to_markdown(tablefmt='pipe', index=False)
        x = re.sub("\|\:\-.*\-\-\|", '', x).replace('\n\n', '\n')
        template = templates.opt_prompt
        prompt = PromptTemplate(template=template,input_variables=['demand','triggerkeyword', 'timeframe', 'optimization_scenario'])
        llm = init_llm(model, max_tokens=1500)
        ### gpt-4 ### gpt-35-turbo
        llm_chain = LLMChain(prompt=prompt,llm=llm)
        response = llm_chain.invoke({'demand':demand, 'triggerkeyword': triggerkeyword,'timeframe': timeframe,'optimization_scenario': x })
        format_response = formatted_json_response(response['text'])
        return {"status" :"Success","userquery" :userquery, "intermediate" : {"input_keywords":prompt.input_variables,"template":prompt.template},"final": response['text'],"format_response":format_response}
    except Exception as err:
        print(err)
        return {"status" :f"Failed - {err}","userquery" :'NA', "intermediate" : {},"final":"NA" }



if __name__ == '__main__':
    app.run(host="0.0.0.0",port=9001)
