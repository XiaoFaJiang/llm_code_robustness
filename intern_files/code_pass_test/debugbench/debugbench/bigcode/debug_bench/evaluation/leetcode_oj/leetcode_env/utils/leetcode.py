import ast
import os
import leetcode

def id_from_slug(slug: str, api_instance) -> str:
    """
    Retrieves the id of the question with the given slug
    """
    graphql_request = leetcode.GraphqlQuery(
      query="""
                  query getQuestionDetail($titleSlug: String!) {
                    question(titleSlug: $titleSlug) {
                      questionId
                    }
                  }
              """,
              variables={"titleSlug": slug},
              operation_name="getQuestionDetail",
      )
    time = 0
    while time < 10:
        try:
            response = ast.literal_eval(str(api_instance.graphql_post(body=graphql_request)))
            break
        except Exception as e:
            print(f"请求Leetcode失败, 重试中({time}/10):{e}")
            time += 1
            if time >= 10:
                raise e

    frontend_id = response['data']['question']['question_id']
    print(f"frontend_id: {frontend_id}")
    return frontend_id

def metadata_from_slug(slug: str, api_instance) -> str:
    """
    Retrieves the metadata of the question with the given slug
    """
    os.environ['HTTP_PROXY'] = 'http://10.253.34.172:6666'
    os.environ['HTTPS_PROXY'] = 'http://10.253.34.172:6666'
    graphql_request = leetcode.GraphqlQuery(
      query="""
                  query getQuestionDetail($titleSlug: String!) {
                    question(titleSlug: $titleSlug) {
                      metaData
                    }
                  }
              """,
              variables={"titleSlug": slug},
              operation_name="getQuestionDetail",
    )
    response = ast.literal_eval(str(api_instance.graphql_post(body=graphql_request)))
    metadata = response['data']['question']
    return metadata
