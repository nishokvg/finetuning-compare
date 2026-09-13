# Account-creation label audit

The supplied smoke test expects **Active Directory** for a generic new-user account request. The trained/merged model predicted **Fileservice**. Word-TFIDF retrieval from the training split found these related examples:

| Training ticket ID | Supplied label | Text |
|---|---|---|
| f6540dfb5fb49d00 | Fileservice | Hi, could you please create a new user account for me? Thanks so much for your help! |
| 0708558895fa8c32 | Support general | Could you please create a new Windows user account for a new starter? They'll need Exchange email, Skype, and the standard file share access set up. |
| 20196e30b1444f4a | Support general | Hi, could you please create a new user account for someone who's joining us? Let me know if you need any details from me to get it set up. |
| 0ba1afcbcfe4adb4 | Active Directory | Hi, we have a new employee starting soon and need to get them set up with a Windows account, email, Skype, and access to the standard file shares. Could you help get that all created? Thanks! |
| 9af30442d900f878 | Active Directory | Hi, I need a new user account created for a new employee with username [USERNAME], including corporate network access and an email account. This is for location ID 13413 — [LOCATION]. Please let me know if you need anything else. |
| 8085031de7fa3b03 | Support general | Hey, could you create a new Windows user account for a new starter? They'll need Exchange email, Skype, and standard file share access set up. Appreciate it! |

The generic request occurs under Fileservice and Support general, while account-onboarding requests with network/email access also occur under Active Directory. This suggests inconsistent annotation or missing routing-policy context; it is not proof of one causal explanation for the model error. No labels were changed.

Next improvement: establish category definitions, review ambiguous account-creation and software-installation tickets, and apply the same revised dataset to both platforms. Do not relabel only the final-test mistakes or claim the existing test remains untouched after using it to tune the next model.

The higher-precision export was chosen using validation. Its final test accuracy and macro-F1 matched the adapter. The first 4-bit export is retained to demonstrate why a five-ticket smoke test cannot replace full evaluation.
