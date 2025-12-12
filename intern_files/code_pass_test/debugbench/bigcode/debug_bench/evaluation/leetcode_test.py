import os
cmd= """curl 'https://leetcode.com/problems/add-two-numbers/submit/' \
  -H 'authority: leetcode.com' \
  -H 'accept: */*' \
  -H 'accept-language: zh-CN,zh;q=0.9,en;q=0.8' \
  -H 'content-type: application/json' \
  -H 'cookie: _gid=GA1.2.128221549.1710244354; gr_user_id=3636eb64-6836-4557-a7b0-e57307634814; csrftoken=ybm3rOpbmIkmH9Br8hQpUGiAl0RTKBY2o0JsYfmWG44WephKUbDwQjYSpkOHvycG; messages=W1siX19qc29uX21lc3NhZ2UiLDAsMjUsIlN1Y2Nlc3NmdWxseSBzaWduZWQgaW4gYXMgQ29yc2t5LiJdXQ:1rkEIC:8bBKpbxtND3YBiUGiRrXR9m5fwV0W4jBef_vDbFlQMs; 87b5a3c3f1a55520_gr_last_sent_cs1=Corsky; INGRESSCOOKIE=27543b69231615b185891654a9c89d05|8e0876c7c1464cc0ac96bc2edceabd27; __stripe_mid=c7146692-1862-450c-906f-68493dd64f5e59f659; __gads=ID=909c24e6b6f45eb4:T=1710296677:RT=1710332309:S=ALNI_MZU3vtlFg3rptG_mR9RKQRaxR33kQ; __gpi=UID=00000d357faed9d7:T=1710296677:RT=1710332309:S=ALNI_MYQFiIdKaIWdYzwMW0V1uOsCJqf4Q; __eoi=ID=c189fe46baeb45a9:T=1710296677:RT=1710332309:S=AA-AfjaFDgzEJJaz5VdDOmrvJW4S; FCNEC=%5B%5B%22AKsRol-nQMRvy_ZSSzmKDc40xQ1eP1gusp2mWujqt7Cf5n5wqd3-mygKN5JwcP3p0AOHr-PObVM0F0fi8h8sBjEcrgFWKfAzXv1Na9Op0fiUZ2fGZT1zizqxG90uFQmkz1Fs71zC-wxZdobvWcmi3yfMKH38eUobHA%3D%3D%22%5D%5D; 87b5a3c3f1a55520_gr_session_id=400279b0-d5cb-494b-b58c-4aff300fdd14; 87b5a3c3f1a55520_gr_last_sent_sid_with_cs1=400279b0-d5cb-494b-b58c-4aff300fdd14; 87b5a3c3f1a55520_gr_session_id_sent_vst=400279b0-d5cb-494b-b58c-4aff300fdd14; cf_clearance=HlFlgXCLP1lzNhrpFFbhGzl9d7NWiVMA2JulyyvGxdY-1710382633-1.0.1.1-gDQI3y2YNSF2VLgJyrTpsLSf0USuGH_.nKcUulSU4NdmwuTlYPmlKLGhX4gPn3S1w3A1qJvvpMJIiIqIGAgzbg; LEETCODE_SESSION=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJfYXV0aF91c2VyX2lkIjoiODU5NTQyIiwiX2F1dGhfdXNlcl9iYWNrZW5kIjoiZGphbmdvLmNvbnRyaWIuYXV0aC5iYWNrZW5kcy5Nb2RlbEJhY2tlbmQiLCJfYXV0aF91c2VyX2hhc2giOiI5Mzk0YTVmNjNjNDY4ZWZhZGU3ZGE3OWFhODcwYjM2OTg1Yjk3ZDhmZjczZjIxMGRkMWQ1MTk0OTdlNWExZGYxIiwiaWQiOjg1OTU0MiwiZW1haWwiOiJ5a3poYW85N0Bnd3UuZWR1IiwidXNlcm5hbWUiOiJDb3Jza3kiLCJ1c2VyX3NsdWciOiJDb3Jza3kiLCJhdmF0YXIiOiJodHRwczovL2Fzc2V0cy5sZWV0Y29kZS5jb20vdXNlcnMvZGVmYXVsdF9hdmF0YXIuanBnIiwicmVmcmVzaGVkX2F0IjoxNzEwMzg0MTcwLCJpcCI6IjEwMy4yMDIuMTQ3LjE5NSIsImlkZW50aXR5IjoiZGQ3ODg3OGJlYmMwZTZhZmZmODBiZTk2NTE2NTExZDciLCJzZXNzaW9uX2lkIjo1NzQ4MzIwNSwiX3Nlc3Npb25fZXhwaXJ5IjoxMjA5NjAwfQ.8RW6OuzaqlP8yVqx2ihXGgoikGi2ayupETcAcSASeWE; __cf_bm=diPiBBp_jlgcptZrpNaOqkxdZlwTkIIlCbEscQftyjk-1710384170-1.0.1.1-blzz6gE2FVDaVjRepwhbRFITOscxnIQX1Tmx6abNVDmuAuJTr0B9EQpZFnukEBoXPa301_Pw.iRdjABmdn1b2Q; _gat=1; 87b5a3c3f1a55520_gr_cs1=Corsky; _ga=GA1.1.1078447927.1704698347; _ga_CDRWKZTDEX=GS1.1.1710382629.8.1.1710384962.49.0.0' \
  -H 'origin: https://leetcode.com' \
  -H 'referer: https://leetcode.com/problems/add-two-numbers/' \
  -H 'sec-ch-ua: "Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"' \
  -H 'sec-ch-ua-mobile: ?0' \
  -H 'sec-ch-ua-platform: "macOS"' \
  -H 'sec-fetch-dest: empty' \
  -H 'sec-fetch-mode: cors' \
  -H 'sec-fetch-site: same-origin' \
  -H 'user-agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36' \
  -H 'x-csrftoken: ybm3rOpbmIkmH9Br8hQpUGiAl0RTKBY2o0JsYfmWG44WephKUbDwQjYSpkOHvycG' \
  --data-raw '{"lang":"python","question_id":"2","typed_code":"# Definition for singly-linked list.\n# class ListNode(object):\n#     def __init__(self, val=0, next=None):\n#         self.val = val\n#         self.next = next\nclass Solution(object):\n    def addTwoNumbers(self, l1, l2):\n        \"\"\"\n        :type l1: ListNode\n        :type l2: ListNode\n        :rtype: ListNode\n        \"\"\"\n        "}'"""

os.system(cmd)