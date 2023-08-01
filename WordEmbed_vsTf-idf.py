###############################################################################################################################################
# print()
# print('**** most_similarity..................')
# most_similatiry = []
# for word in all_abstract_Words:
#
# #     print("word: ", word)
# #
# #     most_similatiry.append(model.wv.most_similar(word))
# #     print(model.wv.most_similar(word)," ", end = " ")
# #     print(" ")
# # print(most_similatiry)
# most_df = pd.DataFrame(data=most_similatiry)
# print(most_df)
#
#
#
# for word_1 in all_abstract_Words:
#     for word_2 in all_abstract_Words:
#         print((word_1, word_2), " : ", end="")
#     print("")
#
#
#
# print()
# print("***Distance Matrix................")
# distence = []
# for word_1 in all_abstract_Words:
#     line = []
#     for word_2 in all_abstract_Words:
#         line.append(model.wv.distance(word_1, word_2))
#         #print("word: ", word_1, word_2)
#     distence.append(line)
# dis_df = pd.DataFrame(data=distence)
# print(dis_df)
#
#
# ##Document-term-matrix........................................................
# print()
# print("***Term-Document matrix................")
# term_doc_matrix = {}
# for term in all_abstract_Words:
#     term_doc_matrix[term] = []
#
#     for doc in all_abstract:
#         if term in doc:
#             term_doc_matrix[term].append(1)
#         else:
#             term_doc_matrix[term].append(0)
# df_1 = pd.DataFrame(data=term_doc_matrix)
# df_1.to_excel('term_doc_matrix.xlsx')
# print(df_1)
#
#
# # ##weight matrix....................................................................
# print()
# print("***CBOW_10-words_weight matrix.......................")
# weight_matrix = []
# for i in all_abstract_Words[:10]:
#     line = []
#     for j in all_abstract_Words:
#         line.append(model.wv.similarity(i, j))
#     weight_matrix.append(line)
#     df_cbow10 = pd.DataFrame(data=weight_matrix)
# #print(df_cbow150)
#
#
# # #Document Term Matrix & Weight Matrix...................................................
# print()
# print("***Document Term Matrix & Weight Matrix scroing........")
# df_2 = np.dot(df_1, np.asarray(df_cbow10.transpose()))
# df_3 = np.savetxt('cbow10_score.out', df_2,  delimiter=',  ')
# df_3 = np.shape(df_2)
# #print(df_3)
#
#
# print()
# print("***CBOW_150-words_weight matrix.......................")
# weight_matrix = []
# for i in all_abstract_Words[:20]:
#     line = []
#     for j in all_abstract_Words:
#         line.append(model.wv.similarity(i, j))
#     weight_matrix.append(line)
#     df_cbow20 = pd.DataFrame(data=weight_matrix)
# #print(df_cbow150)
#
#
# # #Document Term Matrix & Weight Matrix...................................................
# print()
# print("***Document Term Matrix & Weight Matrix scroing........")
# df_2 = np.dot(df_1, np.asarray(df_cbow20.transpose()))
# df_3 = np.savetxt('cbow20_score.out', df_2,  delimiter=',  ')
# df_3 = np.shape(df_2)
# print(df_3)
#
#
#
# print()
# print("******CBOW_30weight matrix.......................")
# weight_matrix = []
# for i in all_abstract_Words[:30]:
#     line = []
#     for j in all_abstract_Words:
#         line.append(model.wv.similarity(i, j))
#     weight_matrix.append(line)
#     df_cbow30 = pd.DataFrame(data=weight_matrix)
# #print(df_cbow30)
#
#
# # #Document Term Matrix & Weight Matrix...................................................
# print()
# print("***Document Term Matrix & Weight Matrix scroing........")
# df_2 = np.dot(df_1, np.asarray(df_cbow30.transpose()))
# df_3 = np.savetxt('cbow30_score.out', df_2,  delimiter=',  ')
# df_3 = np.shape(df_2)
# print(df_3)
#
#
# print()
# print("******CBOW_60weight matrix.......................")
# weight_matrix = []
# for i in all_abstract_Words[:40]:
#     line = []
#     for j in all_abstract_Words:
#         line.append(model.wv.similarity(i, j))
#     weight_matrix.append(line)
#     df_cbow40 = pd.DataFrame(data=weight_matrix)
# #print(df_cbow60)
#
#
# # #Document Term Matrix & Weight Matrix...................................................
# print()
# print("***Document Term Matrix & Weight Matrix scroing........")
# df_2 = np.dot(df_1, np.asarray(df_cbow40.transpose()))
# df_3 = np.savetxt('cbow40_score.out', df_2,  delimiter=',  ')
# df_3 = np.shape(df_2)
# print(df_3)
#
#
# # ##weight matrix....................................................................
# print()
# print("***CBOW_90-words_weight matrix.......................")
# weight_matrix = []
# for i in all_abstract_Words[:50]:
#     line = []
#     for j in all_abstract_Words:
#         line.append(model.wv.similarity(i, j))
#     weight_matrix.append(line)
#     df_cbow50 = pd.DataFrame(data=weight_matrix)
# #print(df_cbow90)
#
#
# # #Document Term Matrix & Weight Matrix...................................................
# print()
# print("***Document Term Matrix & Weight Matrix scroing........")
# df_2 = np.dot(df_1, np.asarray(df_cbow50.transpose()))
# df_3 = np.savetxt('cbow50_score.out', df_2,  delimiter=',  ')
# df_3 = np.shape(df_2)
# print(df_3)
#
# #
# # print()
# # print("***CBOW_120-words_weight matrix.......................")
# # weight_matrix = []
# # for i in all_abstract_Words[:120]:
# #     line = []
# #     for j in all_abstract_Words:
# #         #print((j, i), " : ", end=" ")
# #         line.append(model.wv.similarity(i, j))
# #     weight_matrix.append(line)
# #     df_cbow120 = pd.DataFrame(data=weight_matrix)
# # #print(df_cbow120)
# #
# #
# # # #Document Term Matrix & Weight Matrix...................................................
# # print()
# # print("***Document Term Matrix & Weight Matrix scroing........")
# # df_2 = np.dot(df_1, np.asarray(df_cbow120.transpose()))
# # df_3 = np.savetxt('cbow120_score.out', df_2,  delimiter=',  ')
# # df_3 = np.shape(df_2)
# # print(df_3)
# #
# #
# # print()
# # print("***CBOW_150-words_weight matrix.......................")
# # weight_matrix = []
# # for i in all_abstract_Words[:150]:
# #     line = []
# #     for j in all_abstract_Words:
# #         line.append(model.wv.similarity(i, j))
# #     weight_matrix.append(line)
# #     df_cbow150 = pd.DataFrame(data=weight_matrix)
# # #print(df_cbow150)
# #
# #
# # # #Document Term Matrix & Weight Matrix...................................................
# # print()
# # print("***Document Term Matrix & Weight Matrix scroing........")
# # df_2 = np.dot(df_1, np.asarray(df_cbow150.transpose()))
# # df_3 = np.savetxt('cbow150_score.out', df_2,  delimiter=',  ')
# # df_3 = np.shape(df_2)
# # print(df_3)
# #
#
#
#
# ###Skip-gram..........................................................................
# model = Word2Vec(sentences=all_abstract, size=100, window=5, min_count=50, workers=4, iter=100, sg=1)
# model.wv.save_word2vec_format('sg-100-dimension.txt', binary=False)
#
#
# maximum = len(model.wv.vocab)
# all_abstract_Words = []
# for i in range(0, maximum):
#     all_abstract_Words.append(model.wv.index2word[i])
#
# all_abstract_Words.remove('paper')
# all_abstract_Words.remove('article')
# all_abstract_Words.remove('approach')
# all_abstract_Words.remove('performance')
#
#
#
# # print()
# # print('**** most_similarity..................')
# # most_similatiry = []
# # for word in all_abstract_Words:
# #
# # #     print("word: ", word)
# # #
# # #     most_similatiry.append(model.wv.most_similar(word))
# # #     print(model.wv.most_similar(word)," ", end = " ")
# # #     print(" ")
# # # print(most_similatiry)
# # most_df = pd.DataFrame(data=most_similatiry)
# # print(most_df)
#
#
# #
# # for word_1 in all_abstract_Words:
# #     for word_2 in all_abstract_Words:
# #         print((word_1, word_2), " : ", end="")
# #     print("")
#
#
#
# print()
# print("***SG_Distance Matrix................")
# distence = []
# for word_1 in all_abstract_Words:
#     line = []
#     for word_2 in all_abstract_Words:
#         line.append(model.wv.distance(word_1, word_2))
#         #print("word: ", word_1, word_2)
#     distence.append(line)
# dis_df = pd.DataFrame(data=distence)
# print(dis_df)
#
#
# ##Document-term-matrix........................................................
# print()
# print("***SG_DTM................")
# term_doc_matrix = {}
# for term in all_abstract_Words:
#     term_doc_matrix[term] = []
#
#     for doc in all_abstract:
#         if term in doc:
#             term_doc_matrix[term].append(1)
#         else:
#             term_doc_matrix[term].append(0)
# df_1 = pd.DataFrame(data=term_doc_matrix)
# #df_1.to_excel('term_doc_matrix.xlsx')
# print(df_1)
#
#
# # # ##weight matrix....................................................................
# #
# # print()
# # print("***SG_weight matrix.......................")
# # weight_matrix = []
# # for i in all_abstract_Words[:10]:
# #     line = []
# #     for j in all_abstract_Words:
# #         #print((j, i), " : ", end=" ")
# #         line.append(model.wv.similarity(i, j))
# #     weight_matrix.append(line)
# #     df = pd.DataFrame(data=weight_matrix)
# # print(df)
# # #
# # #
# # # #Document Term Matrix & Weight Matrix...................................................
# # print()
# # print("***SG_DTM & Weight Matrix scoring........")
# # df_2 = np.dot(df_1, np.asarray(df.transpose()))
# # df_3 = np.savetxt('sg_score.out', df_2,  delimiter=',  ')
# # df_3 = np.shape(df_2)
# # print(df_3)
#
#
#
# print()
# print("***SG_10_weight matrix.......................")
# weight_matrix = []
# for i in all_abstract_Words[:10]:
#     line = []
#     for j in all_abstract_Words:
#         line.append(model.wv.similarity(i, j))
#     weight_matrix.append(line)
#     df_10 = pd.DataFrame(data=weight_matrix)
# #print(df_120)
#
#
# # #Document Term Matrix & Weight Matrix...................................................
# print()
# print("***SG_DTM & Weight Matrix scoring........")
# df_2 = np.dot(df_1, np.asarray(df_10.transpose()))
# df_3 = np.savetxt('sg_10_score.out', df_2,  delimiter=',  ')
# df_3 = np.shape(df_2)
# #print(df_3)
#
#
# print()
# print("***SG_20_weight matrix.......................")
# weight_matrix = []
# for i in all_abstract_Words[:20]:
#     line = []
#     for j in all_abstract_Words:
#         line.append(model.wv.similarity(i, j))
#     weight_matrix.append(line)
#     df_20 = pd.DataFrame(data=weight_matrix)
# #print(df_120)
#
#
# # #Document Term Matrix & Weight Matrix...................................................
# print()
# print("***SG_DTM & Weight Matrix scoring........")
# df_2 = np.dot(df_1, np.asarray(df_20.transpose()))
# df_3 = np.savetxt('sg_20_score.out', df_2,  delimiter=',  ')
# df_3 = np.shape(df_2)
# print(df_3)
#
# ## words multipication............................................
# print()
# print("***SG_30_weight matrix.......................")
# weight_matrix = []
# for i in all_abstract_Words[:30]:
#     line = []
#     for j in all_abstract_Words:
#         line.append(model.wv.similarity(i, j))
#     weight_matrix.append(line)
#     df_30 = pd.DataFrame(data=weight_matrix)
# #print(df_30)
#
#
# # #Document Term Matrix & Weight Matrix...................................................
# print()
# print("***SG_DTM & Weight Matrix scoring........")
# df_2 = np.dot(df_1, np.asarray(df_30.transpose()))
# df_3 = np.savetxt('sg_30_score.out', df_2,  delimiter=',  ')
# df_3 = np.shape(df_2)
# print(df_3)
#
#
#
# print()
# print("***SG_60_weight matrix.......................")
# weight_matrix = []
# for i in all_abstract_Words[:40]:
#     line = []
#     for j in all_abstract_Words:
#         line.append(model.wv.similarity(i, j))
#     weight_matrix.append(line)
#     df_40 = pd.DataFrame(data=weight_matrix)
# #print(df650)
# #
# #
# # #Document Term Matrix & Weight Matrix...................................................
# print()
# print("***SG_DTM & Weight Matrix scoring........")
# df_2 = np.dot(df_1, np.asarray(df_40.transpose()))
# df_3 = np.savetxt('sg_40_score.out', df_2,  delimiter=',  ')
# df_3 = np.shape(df_2)
# print(df_3)
#
#
#
# print()
# print("***SG_50_weight matrix.......................")
# weight_matrix = []
# for i in all_abstract_Words[:50]:
#     line = []
#     for j in all_abstract_Words:
#         line.append(model.wv.similarity(i, j))
#     weight_matrix.append(line)
#     df_50 = pd.DataFrame(data=weight_matrix)
# #print(df_90)
#
#
# # #Document Term Matrix & Weight Matrix...................................................
# print()
# print("***SG_DTM & Weight Matrix scoring........")
# df_2 = np.dot(df_1, np.asarray(df_50.transpose()))
# df_3 = np.savetxt('sg_50_score.out', df_2,  delimiter=',  ')
# df_3 = np.shape(df_2)
# print(df_3)
#
# #
# # print()
# # print("***SG_120_weight matrix.......................")
# # weight_matrix = []
# # for i in all_abstract_Words[:120]:
# #     line = []
# #     for j in all_abstract_Words:
# #         line.append(model.wv.similarity(i, j))
# #     weight_matrix.append(line)
# #     df_120 = pd.DataFrame(data=weight_matrix)
# # #print(df_120)
# #
# #
# # # #Document Term Matrix & Weight Matrix...................................................
# # print()
# # print("***SG_DTM & Weight Matrix scoring........")
# # df_2 = np.dot(df_1, np.asarray(df_120.transpose()))
# # df_3 = np.savetxt('sg_120_score.out', df_2,  delimiter=',  ')
# # df_3 = np.shape(df_2)
# # print(df_3)
# #
# #
# # print()
# # print("***SG_150_weight matrix.......................")
# # weight_matrix = []
# # for i in all_abstract_Words[:150]:
# #     line = []
# #     for j in all_abstract_Words:
# #         line.append(model.wv.similarity(i, j))
# #     weight_matrix.append(line)
# #     df_150 = pd.DataFrame(data=weight_matrix)
# # #print(df_120)
# #
# #
# # # #Document Term Matrix & Weight Matrix...................................................
# # print()
# # print("***SG_DTM & Weight Matrix scoring........")
# # df_2 = np.dot(df_1, np.asarray(df_150.transpose()))
# # df_3 = np.savetxt('sg_150_score.out', df_2,  delimiter=',  ')
# # df_3 = np.shape(df_2)
# # print(df_3)
# #
#
#
#
# # ## TF-IDF.................................................................................
# N = len(all_abstract)
#
#
# def tf(words, docs):
#     return docs.count(words)
#
# def idf(words):
#     df = 0
#     for docs in all_abstract:
#         df += words in docs
#     return log(N/(df + 1))
#
#
# def tfidf(words, docs):
#     return tf(words, docs) * idf(words)
#
#
# # tf_result = []
# # for i in range(N):
# #     tf_result.append([])
# #     docs = all_abstract[i]
# #     for j in range(len(all_abstract_Words)):
# #         words = all_abstract_Words[j]
# #         tf_result[-1].append(tf(words, docs))
# #
# # tf_df = pd.DataFrame(tf_result, columns=all_abstract_Words)
# # print('tf_results...........................................................')
# # print(tf_df)
# #
# #
# # idf_result = []
# # for i in range(len(all_abstract_Words)):
# #     words = all_abstract_Words[i]
# #     idf_result.append(idf(words))
# # idf_df = pd.DataFrame(idf_result, index=all_abstract_Words, columns=["IDF"])
# # print('idf_results.........................................................')
# # print(idf_df)
# #
# #
# # tfidf_result = []
# # for i in range(N):
# #     tfidf_result.append([])
# #     docs = all_abstract[i]
# #     for j in range(len(all_abstract_Words)):
# #         words = all_abstract_Words[j]
# #
# #         tfidf_result[-1].append(tfidf(words, docs))
# #
# # tfidf_df = pd.DataFrame(tfidf_result, columns=all_abstract_Words)
# # tfidf_df.to_excel('tfidf_result.xlsx')
# # print('tfidf_results............................................')
# # print(tfidf_df)
# #
#
#
# ##150-words for tf_idf.....................................................
# print()
# tfidf_result_10 = []
# tfidf_result_20 = []
# tfidf_result_30 = []
# tfidf_result_40 = []
# tfidf_result_50 = []
#
# for i in range(N):
#     tfidf_result_10.append([])
#     tfidf_result_20.append([])
#     tfidf_result_30.append([])
#     tfidf_result_40.append([])
#     tfidf_result_50.append([])
#     docs = all_abstract[i]
#     for words10 in all_abstract_Words[:10]:
#         tfidf_result_10[-1].append(tfidf(words10, docs))
#     for words20 in all_abstract_Words[:20]:
#         tfidf_result_20[-1].append(tfidf(words20, docs))
#     for words30 in all_abstract_Words[:30]:
#         tfidf_result_30[-1].append(tfidf(words30, docs))
#     for words40 in all_abstract_Words[:40]:
#         tfidf_result_40[-1].append(tfidf(words40, docs))
#     for words50 in all_abstract_Words[:50]:
#         tfidf_result_50[-1].append(tfidf(words50, docs))
#
# tfidf_df_1 = pd.DataFrame(tfidf_result_10, columns=all_abstract_Words[:10])
# tfidf_df_1_10 = np.savetxt('tfidf_10_score.out', tfidf_df_1,  delimiter=',  ')
# tfidf_df_2 = pd.DataFrame(tfidf_result_20, columns=all_abstract_Words[:20])
# tfidf_df_2_20 = np.savetxt('tfidf_20_score.out', tfidf_df_2,  delimiter=',  ')
# tfidf_df_3 = pd.DataFrame(tfidf_result_30, columns=all_abstract_Words[:30])
# tfidf_df_3_30 = np.savetxt('tfidf_30_score.out', tfidf_df_3,  delimiter=',  ')
# tfidf_df_4 = pd.DataFrame(tfidf_result_40, columns=all_abstract_Words[:40])
# tfidf_df_4_40 = np.savetxt('tfidf_40_score.out', tfidf_df_4,  delimiter=',  ')
# tfidf_df_5 = pd.DataFrame(tfidf_result_50, columns=all_abstract_Words[:50])
# tfidf_df_5_50 = np.savetxt('tfidf_50_score.out', tfidf_df_5,  delimiter=',  ')
# #print(tfidf_df_1)
#
#
#
#
#
#
#
# #
# #
# # ##150-words for tf_idf.....................................................
# # print()
# # tfidf_result_1 = []
# # tfidf_result_2 = []
# # tfidf_result_3 = []
# # tfidf_result_4 = []
# # tfidf_result_5 = []
# #
# # for i in range(N):
# #     tfidf_result_1.append([])
# #     tfidf_result_2.append([])
# #     tfidf_result_3.append([])
# #     tfidf_result_4.append([])
# #     tfidf_result_5.append([])
# #     docs = all_abstract[i]
# #     for words30 in all_abstract_Words[:30]:
# #         tfidf_result_1[-1].append(tfidf(words30, docs))
# #     for words60 in all_abstract_Words[:60]:
# #         tfidf_result_2[-1].append(tfidf(words60, docs))
# #     for words90 in all_abstract_Words[:90]:
# #         tfidf_result_3[-1].append(tfidf(words90, docs))
# #     for words120 in all_abstract_Words[:120]:
# #         tfidf_result_4[-1].append(tfidf(words120, docs))
# #     for words150 in all_abstract_Words[:150]:
# #         tfidf_result_5[-1].append(tfidf(words150, docs))
# #
# # tfidf_df_1 = pd.DataFrame(tfidf_result_1, columns=all_abstract_Words[:30])
# # tfidf_df_1_30 = np.savetxt('tfidf_30_score.out', tfidf_df_1,  delimiter=',  ')
# # tfidf_df_2 = pd.DataFrame(tfidf_result_2, columns=all_abstract_Words[:60])
# # tfidf_df_2_60 = np.savetxt('tfidf_60_score.out', tfidf_df_2,  delimiter=',  ')
# # tfidf_df_3 = pd.DataFrame(tfidf_result_3, columns=all_abstract_Words[:90])
# # tfidf_df_3_90 = np.savetxt('tfidf_90_score.out', tfidf_df_3,  delimiter=',  ')
# # tfidf_df_4 = pd.DataFrame(tfidf_result_4, columns=all_abstract_Words[:120])
# # tfidf_df_4_120 = np.savetxt('tfidf_120_score.out', tfidf_df_4,  delimiter=',  ')
# # tfidf_df_5 = pd.DataFrame(tfidf_result_5, columns=all_abstract_Words[:150])
# # tfidf_df_5_150 = np.savetxt('tfidf_150_score.out', tfidf_df_5,  delimiter=',  ')
# # # print('tfidf_30-words_results............................................')
# # # print(tfidf_df_1)
# # # print('tfidf_60-words_results............................................')
# # # print(tfidf_df_2)
# # # print('tfidf_90-words_results............................................')
# # # print(tfidf_df_3)
# # # print('tfidf_120-words_results............................................')
# # # print(tfidf_df_4)
# # # print('tfidf_150-words_results............................................')
# # # print(tfidf_df_5)
#
#
#
#
# ### Keyword data training for relevant documents.......................
# with open('ReadTexT.txt', 'rt', encoding='UTF8') as file:
#     all_keyword = []
#     for line in file:
#         if '<keyword>' in line:
#             keyword = line.split('</keyword>')[0].split('<keyword>')[-1]
#             keyword = ''.join(line for line in keyword if not keyword.isdigit())
#             keyword = regex.sub('[^\w\d\s]+', '', keyword)
#             keyword = nltk.word_tokenize(keyword)
#             stop_words = set(stopwords.words('english'))
#             filtered_sentence_keyword = [w.lower() for w in keyword if
#                                          w.lower() not in punctuation and w.lower() not in stop_words]
#             tagged_list = nltk.pos_tag(filtered_sentence_keyword)
#             nouns_list = [t[0] for t in tagged_list if t[-1] == 'NN']
#             lm = WordNetLemmatizer()
#             singluar_form = [lm.lemmatize(w, pos='v') for w in nouns_list]
#             all_keyword.append(singluar_form)
#
# #print(all_keyword)
#
# ##CBOW evaluation..............................................................................................................................
# ### CBOW10-words_Retrieved document data
# line_df1 = []
# with open("cbow10_score.out", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line_list = line.replace(',', ' ').replace('\n', ' ').split()
#         line_df = []
#         for num_str in line_list:
#             num = float(num_str)
#             line_df.append(num)
#         line_df1.append(line_df)
#         #print('line_df1:', line_df1)
#
#     all_line = []
#     one_line = []
#     count = 0
#     for i in line_df1:
#         one_line = []
#         Min = min(i)
#         Max = max(i)
#         for j in i:
#             one_line.append((j - Min)/(Max - Min))
#         all_line.append(one_line)
#     #print('all_line')
#     #print(all_line)
# #retrieved_df = pd.DataFrame(data=line_df1)
# # retrieved_df.to_excel('retrievened_documents.xlsx')
# # print('retrieved document')
# # print(str(retrieved_df))
# #
#
# # # # relevant and retrivent data...............................
# count = 0
# score1 = []
# intersection = []
# retrieved_documents = []
# top_keyword = all_abstract_Words[0:10]
# for i in range(0, 1000):
#     score = []
#     retrieved_score = []
#
#     for j in range(0, len(all_line[i])):
#         fl_data = float(all_line[i][j])
#         if fl_data > 0.5 and top_keyword[j] in all_keyword[i]:
#             score.append(1)
#         else:
#             score.append(0)
#
#         if fl_data > 0.5:
#             retrieved_score.append(1)
#         else:
#             retrieved_score.append(0)
#     intersection.append(score)
#     retrieved_documents.append(retrieved_score)
# #print(retrieved_documents)
# #intersection_df = pd.DataFrame(data=intersection)
# # print('intersection_df')
# # print(str(intersection_df))
#
#
# #print("relevant")
# relevant = []
# for i in all_keyword:
#     document_keyword = list(set(i))
#     relevant.append(len(list(set(i))))
# #print(relevant)
#
#
# #print("retrieved")
# retrieved = []
# for i in retrieved_documents:
#     retrieved.append(sum(i))
# #print(retrieved)
#
#
# Recall = []
# for i in range(0, 1000):
#     if relevant[i] == 0:
#         Recall.append(0)
#     else:
#         Recall.append(float(sum(intersection[i]) / relevant[i]))
#
# #Recall_df = pd.DataFrame(data=Recall)
# # print("Recall")
# # print(str(Recall_df))
# # print("Recall average:", max(Recall), sum(Recall)/1000.0)
#
#
# Precision = []
# for i in range(0, 1000):
#     if retrieved[i] == 0:
#         Precision.append(0)
#     else:
#         Precision.append(float(sum(intersection[i]) / retrieved[i]))
#
#
# #Precision_df = pd.DataFrame(data=Precision)
# # print("Precision")
# # print(str(Precision_df))
# # print("precision average:", max(Precision), sum(Precision)/1000.0)
#
# F_score = []
# for i in range(0, 1000):
#     if (Precision[i] + Recall[i]) == 0:
#         F_score.append(0.0)
#     else:
#         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
#
# #print(F_score)
# #F_score_df = pd.DataFrame(data=F_score)
# #print(str(F_score_df))
# F_aver_cbow_10 = (sum(F_score)/1000.0)
# print("cbow_10:", F_aver_cbow_10)
#
#
#
# ### CBOW30-words_Retrieved document data
# line_df1 = []
# with open("cbow20_score.out", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line_list = line.replace(',', ' ').replace('\n', ' ').split()
#         line_df = []
#         for num_str in line_list:
#             num = float(num_str)
#             line_df.append(num)
#         line_df1.append(line_df)
#         #print('line_df1:', line_df1)
#
#     all_line = []
#     one_line = []
#     count = 0
#     for i in line_df1:
#         one_line = []
#         Min = min(i)
#         Max = max(i)
#         for j in i:
#             one_line.append((j - Min)/(Max - Min))
#         all_line.append(one_line)
#     #print('all_line')
#     #print(all_line)
# #retrieved_df = pd.DataFrame(data=line_df1)
# # retrieved_df.to_excel('retrievened_documents.xlsx')
# # print('retrieved document')
# # print(str(retrieved_df))
# #
#
# # # # relevant and retrivent data...............................
# count = 0
# score1 = []
# intersection = []
# retrieved_documents = []
# top_keyword = all_abstract_Words[0:20]
# for i in range(0, 1000):
#     score = []
#     retrieved_score = []
#
#     for j in range(0, len(all_line[i])):
#         fl_data = float(all_line[i][j])
#         if fl_data > 0.5 and top_keyword[j] in all_keyword[i]:
#             score.append(1)
#         else:
#             score.append(0)
#
#         if fl_data > 0.5:
#             retrieved_score.append(1)
#         else:
#             retrieved_score.append(0)
#     intersection.append(score)
#     retrieved_documents.append(retrieved_score)
# #print(retrieved_documents)
# #intersection_df = pd.DataFrame(data=intersection)
# # print('intersection_df')
# # print(str(intersection_df))
#
#
# #print("relevant")
# relevant = []
# for i in all_keyword:
#     document_keyword = list(set(i))
#     relevant.append(len(list(set(i))))
# #print(relevant)
#
#
# #print("retrieved")
# retrieved = []
# for i in retrieved_documents:
#     retrieved.append(sum(i))
# #print(retrieved)
#
#
# Recall = []
# for i in range(0, 1000):
#     if relevant[i] == 0:
#         Recall.append(0)
#     else:
#         Recall.append(float(sum(intersection[i]) / relevant[i]))
#
# #Recall_df = pd.DataFrame(data=Recall)
# # print("Recall")
# # print(str(Recall_df))
# # print("Recall average:", max(Recall), sum(Recall)/1000.0)
#
#
# Precision = []
# for i in range(0, 1000):
#     if retrieved[i] == 0:
#         Precision.append(0)
#     else:
#         Precision.append(float(sum(intersection[i]) / retrieved[i]))
#
#
# #Precision_df = pd.DataFrame(data=Precision)
# # print("Precision")
# # print(str(Precision_df))
# # print("precision average:", max(Precision), sum(Precision)/1000.0)
#
# F_score = []
# for i in range(0, 1000):
#     if (Precision[i] + Recall[i]) == 0:
#         F_score.append(0.0)
#     else:
#         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
#
# #print(F_score)
# #F_score_df = pd.DataFrame(data=F_score)
# #print(str(F_score_df))
# F_aver_cbow_20 = (sum(F_score)/1000.0)
# print("cbow_20:", F_aver_cbow_20)
#
#
#
# ### CBOW30-words_Retrieved document data...............................................................................
# line_df1 = []
# with open("cbow30_score.out", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line_list = line.replace(',', ' ').replace('\n', ' ').split()
#         line_df = []
#         for num_str in line_list:
#             num = float(num_str)
#             line_df.append(num)
#         line_df1.append(line_df)
#         #print('line_df1:', line_df1)
#
#     all_line = []
#     one_line = []
#     count = 0
#     for i in line_df1:
#         one_line = []
#         Min = min(i)
#         Max = max(i)
#         for j in i:
#             one_line.append((j - Min)/(Max - Min))
#         all_line.append(one_line)
#     #print('all_line')
#     #print(all_line)
# #retrieved_df = pd.DataFrame(data=line_df1)
# # retrieved_df.to_excel('retrievened_documents.xlsx')
# # print('retrieved document')
# # print(str(retrieved_df))
# #
#
# # # # relevant and retrivent data...............................
# count = 0
# score1 = []
# intersection = []
# retrieved_documents = []
# top_keyword = all_abstract_Words[0:30]
# for i in range(0, 1000):
#     score = []
#     retrieved_score = []
#
#     for j in range(0, len(all_line[i])):
#         fl_data = float(all_line[i][j])
#         if fl_data > 0.5 and top_keyword[j] in all_keyword[i]:
#             score.append(1)
#         else:
#             score.append(0)
#
#         if fl_data > 0.5:
#             retrieved_score.append(1)
#         else:
#             retrieved_score.append(0)
#     intersection.append(score)
#     retrieved_documents.append(retrieved_score)
# #print(retrieved_documents)
# #intersection_df = pd.DataFrame(data=intersection)
# # print('intersection_df')
# # print(str(intersection_df))
#
#
# #print("relevant")
# relevant = []
# for i in all_keyword:
#     document_keyword = list(set(i))
#     relevant.append(len(list(set(i))))
# #print(relevant)
#
#
# #print("retrieved")
# retrieved = []
# for i in retrieved_documents:
#     retrieved.append(sum(i))
# #print(retrieved)
#
#
# Recall = []
# for i in range(0, 1000):
#     if relevant[i] == 0:
#         Recall.append(0)
#     else:
#         Recall.append(float(sum(intersection[i]) / relevant[i]))
#
# #Recall_df = pd.DataFrame(data=Recall)
# # print("Recall")
# # print(str(Recall_df))
# # print("Recall average:", max(Recall), sum(Recall)/1000.0)
#
#
# Precision = []
# for i in range(0, 1000):
#     if retrieved[i] == 0:
#         Precision.append(0)
#     else:
#         Precision.append(float(sum(intersection[i]) / retrieved[i]))
#
#
# #Precision_df = pd.DataFrame(data=Precision)
# # print("Precision")
# # print(str(Precision_df))
# # print("precision average:", max(Precision), sum(Precision)/1000.0)
#
# F_score = []
# for i in range(0, 1000):
#     if (Precision[i] + Recall[i]) == 0:
#         F_score.append(0.0)
#     else:
#         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
#
# #print(F_score)
# #F_score_df = pd.DataFrame(data=F_score)
# #print(str(F_score_df))
# F_aver_cbow_30 = (sum(F_score)/1000.0)
# print("cbow_30:", F_aver_cbow_30)
#
#
#
#
# ### CBOW60-words_Retrieved document data....................................................................
# line_df1 = []
# with open("cbow40_score.out", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line_list = line.replace(',', ' ').replace('\n', ' ').split()
#         line_df = []
#         for num_str in line_list:
#             num = float(num_str)
#             line_df.append(num)
#         line_df1.append(line_df)
#         #print('line_df1:', line_df1)
#
#     all_line = []
#     one_line = []
#     count = 0
#     for i in line_df1:
#         one_line = []
#         Min = min(i)
#         Max = max(i)
#         for j in i:
#             one_line.append((j - Min)/(Max - Min))
#         all_line.append(one_line)
#     #print('all_line')
#     #print(all_line)
# #retrieved_df = pd.DataFrame(data=line_df1)
# # print('retrieved document')
# # print(str(retrieved_df))
# #
#
# # # # relevant and retrivent data...............................
# count = 0
# score1 = []
# intersection = []
# retrieved_documents = []
# top_keyword = all_abstract_Words[0:60]
# for i in range(0, 1000):
#     score = []
#     retrieved_score = []
#
#     for j in range(0, len(all_line[i])):
#         fl_data = float(all_line[i][j])
#         if fl_data > 0.5 and top_keyword[j] in all_keyword[i]:
#             score.append(1)
#         else:
#             score.append(0)
#
#         if fl_data > 0.5:
#             retrieved_score.append(1)
#         else:
#             retrieved_score.append(0)
#     intersection.append(score)
#     retrieved_documents.append(retrieved_score)
# #intersection_df = pd.DataFrame(data=intersection)
# # print('intersection_df')
# # print(str(intersection_df))
#
#
# #print("relevant")
# relevant = []
# for i in all_keyword:
#     document_keyword = list(set(i))
#     relevant.append(len(list(set(i))))
# #print(relevant)
#
#
# #print("retrieved")
# retrieved = []
# for i in retrieved_documents:
#     retrieved.append(sum(i))
# #print(retrieved)
#
#
# Recall = []
# for i in range(0, 1000):
#     if relevant[i] == 0:
#         Recall.append(0)
#     else:
#         Recall.append(float(sum(intersection[i]) / relevant[i]))
#
# #Recall_df = pd.DataFrame(data=Recall)
# # print("Recall")
# # print(str(Recall_df))
# # print("Recall average:", max(Recall), sum(Recall)/1000.0)
#
#
# Precision = []
# for i in range(0, 1000):
#     if retrieved[i] == 0:
#         Precision.append(0)
#     else:
#         Precision.append(float(sum(intersection[i]) / retrieved[i]))
#
#
# #Precision_df = pd.DataFrame(data=Precision)
# # print("Precision")
# # print(str(Precision_df))
# # print("precision average:", max(Precision), sum(Precision)/1000.0)
#
# F_score = []
# for i in range(0, 1000):
#     if (Precision[i] + Recall[i]) == 0:
#         F_score.append(0.0)
#     else:
#         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
#
# #print(F_score)
# #F_score_df = pd.DataFrame(data=F_score)
# #print("F_score")
# #print(str(F_score_df))
# F_aver_cbow_40 = (sum(F_score)/1000.0)
# print("cbow_40:", F_aver_cbow_40)
#
#
#
#
# ### CBOW90-words_Retrieved document data....................................................................
# line_df1 = []
# with open("cbow50_score.out", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line_list = line.replace(',', ' ').replace('\n', ' ').split()
#         line_df = []
#         for num_str in line_list:
#             num = float(num_str)
#             line_df.append(num)
#         line_df1.append(line_df)
#         #print('line_df1:', line_df1)
#
#     all_line = []
#     one_line = []
#     count = 0
#     for i in line_df1:
#         one_line = []
#         Min = min(i)
#         Max = max(i)
#         for j in i:
#             one_line.append((j - Min)/(Max - Min))
#         all_line.append(one_line)
#     #print('all_line')
#     #print(all_line)
# #retrieved_df = pd.DataFrame(data=line_df1)
# # retrieved_df.to_excel('retrievened_documents.xlsx')
# # print('retrieved document')
# # print(str(retrieved_df))
# #
#
# # # # relevant and retrivent data...............................
# count = 0
# score1 = []
# intersection = []
# retrieved_documents = []
# top_keyword = all_abstract_Words[0:90]
# for i in range(0, 1000):
#     score = []
#     retrieved_score = []
#
#     for j in range(0, len(all_line[i])):
#         fl_data = float(all_line[i][j])
#         if fl_data > 0.5 and top_keyword[j] in all_keyword[i]:
#             score.append(1)
#         else:
#             score.append(0)
#
#         if fl_data > 0.5:
#             retrieved_score.append(1)
#         else:
#             retrieved_score.append(0)
#     intersection.append(score)
#     retrieved_documents.append(retrieved_score)
# #intersection_df = pd.DataFrame(data=intersection)
# # print('intersection_df')
# # print(str(intersection_df))
#
#
# #print("relevant")
# relevant = []
# for i in all_keyword:
#     document_keyword = list(set(i))
#     relevant.append(len(list(set(i))))
# #print(relevant)
#
#
# #print("retrieved")
# retrieved = []
# for i in retrieved_documents:
#     retrieved.append(sum(i))
# #print(retrieved)
#
#
# Recall = []
# for i in range(0, 1000):
#     if relevant[i] == 0:
#         Recall.append(0)
#     else:
#         Recall.append(float(sum(intersection[i]) / relevant[i]))
#
# #Recall_df = pd.DataFrame(data=Recall)
# # print("Recall")
# # print(str(Recall_df))
# # print("Recall average:", max(Recall), sum(Recall)/1000.0)
#
#
# Precision = []
# for i in range(0, 1000):
#     if retrieved[i] == 0:
#         Precision.append(0)
#     else:
#         Precision.append(float(sum(intersection[i]) / retrieved[i]))
#
#
# #Precision_df = pd.DataFrame(data=Precision)
# # print("Precision")
# # print(str(Precision_df))
# # print("precision average:", max(Precision), sum(Precision)/1000.0)
#
# F_score = []
# for i in range(0, 1000):
#     if (Precision[i] + Recall[i]) == 0:
#         F_score.append(0.0)
#     else:
#         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
#
# #print(F_score)
# #F_score_df = pd.DataFrame(data=F_score)
# #print("F_score")
# #print(str(F_score_df))
# F_aver_cbow_50 = (sum(F_score)/1000.0)
# print("cbow_50:", F_aver_cbow_50)
#
#
# #
# #
# # ### CBOW120-words_Retrieved document data....................................................................
# # line_df1 = []
# # with open("cbow120_score.out", 'r') as f:
# #     lines = f.readlines()
# #     for line in lines:
# #         line_list = line.replace(',', ' ').replace('\n', ' ').split()
# #         line_df = []
# #         for num_str in line_list:
# #             num = float(num_str)
# #             line_df.append(num)
# #         line_df1.append(line_df)
# #         #print('line_df1:', line_df1)
# #
# #     all_line = []
# #     one_line = []
# #     count = 0
# #     for i in line_df1:
# #         one_line = []
# #         Min = min(i)
# #         Max = max(i)
# #         for j in i:
# #             one_line.append((j - Min)/(Max - Min))
# #         all_line.append(one_line)
# #     #print('all_line')
# #     #print(all_line)
# # #retrieved_df = pd.DataFrame(data=line_df1)
# # # retrieved_df.to_excel('retrievened_documents.xlsx')
# # # print('retrieved document')
# # # print(str(retrieved_df))
# # #
# #
# # # # # relevant and retrivent data...............................
# # count = 0
# # score1 = []
# # intersection = []
# # retrieved_documents = []
# # top_keyword = all_abstract_Words[0:120]
# # for i in range(0, 1000):
# #     score = []
# #     retrieved_score = []
# #
# #     for j in range(0, len(all_line[i])):
# #         fl_data = float(all_line[i][j])
# #         if fl_data > 0.1 and top_keyword[j] in all_keyword[i]:
# #             score.append(1)
# #         else:
# #             score.append(0)
# #
# #         if fl_data > 0.1:
# #             retrieved_score.append(1)
# #         else:
# #             retrieved_score.append(0)
# #     intersection.append(score)
# #     retrieved_documents.append(retrieved_score)
# # #intersection_df = pd.DataFrame(data=intersection)
# # # print('intersection_df')
# # # print(str(intersection_df))
# #
# #
# # #print("relevant")
# # relevant = []
# # for i in all_keyword:
# #     document_keyword = list(set(i))
# #     relevant.append(len(list(set(i))))
# # #print(relevant)
# #
# #
# # #print("retrieved")
# # retrieved = []
# # for i in retrieved_documents:
# #     retrieved.append(sum(i))
# # #print(retrieved)
# #
# #
# # Recall = []
# # for i in range(0, 1000):
# #     if relevant[i] == 0:
# #         Recall.append(0)
# #     else:
# #         Recall.append(float(sum(intersection[i]) / relevant[i]))
# #
# # #Recall_df = pd.DataFrame(data=Recall)
# # # print("Recall")
# # # print(str(Recall_df))
# # # print("Recall average:", max(Recall), sum(Recall)/1000.0)
# #
# #
# # Precision = []
# # for i in range(0, 1000):
# #     if retrieved[i] == 0:
# #         Precision.append(0)
# #     else:
# #         Precision.append(float(sum(intersection[i]) / retrieved[i]))
# #
# #
# # #Precision_df = pd.DataFrame(data=Precision)
# # # print("Precision")
# # # print(str(Precision_df))
# # # print("precision average:", max(Precision), sum(Precision)/1000.0)
# #
# # F_score = []
# # for i in range(0, 1000):
# #     if (Precision[i] + Recall[i]) == 0:
# #         F_score.append(0.0)
# #     else:
# #         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
# #
# # #print(F_score)
# # #F_score_df = pd.DataFrame(data=F_score)
# # #print("F_score")
# # #print(str(F_score_df))
# # F_aver_cbow_120 = (sum(F_score)/1000.0)
# # print("cbow_120:",F_aver_cbow_120)
# #
# #
# #
# #
# # ### CBOW150-words_Retrieved document data....................................................................
# # line_df1 = []
# # with open("cbow150_score.out", 'r') as f:
# #     lines = f.readlines()
# #     for line in lines:
# #         line_list = line.replace(',', ' ').replace('\n', ' ').split()
# #         line_df = []
# #         for num_str in line_list:
# #             num = float(num_str)
# #             line_df.append(num)
# #         line_df1.append(line_df)
# #         # print('line_df1:', line_df1)
# #
# #     all_line = []
# #     one_line = []
# #     count = 0
# #     for i in line_df1:
# #         one_line = []
# #         Min = min(i)
# #         Max = max(i)
# #         for j in i:
# #             one_line.append((j - Min) / (Max - Min))
# #         all_line.append(one_line)
# #     # print('all_line')
# #     # print(all_line)
# # # retrieved_df = pd.DataFrame(data=line_df1)
# # # print('retrieved document')
# # # print(str(retrieved_df))
# # #
# #
# # # # # relevant and retrivent data...............................
# # count = 0
# # score1 = []
# # intersection = []
# # retrieved_documents = []
# # top_keyword = all_abstract_Words[0:150]
# # for i in range(0, 1000):
# #     score = []
# #     retrieved_score = []
# #
# #     for j in range(0, len(all_line[i])):
# #         fl_data = float(all_line[i][j])
# #         if fl_data > 0.1 and top_keyword[j] in all_keyword[i]:
# #             score.append(1)
# #         else:
# #             score.append(0)
# #
# #         if fl_data > 0.1:
# #             retrieved_score.append(1)
# #         else:
# #             retrieved_score.append(0)
# #     intersection.append(score)
# #     retrieved_documents.append(retrieved_score)
# # #intersection_df = pd.DataFrame(data=intersection)
# # # print('intersection_df')
# # # print(str(intersection_df))
# #
# #
# # #print("relevant")
# # relevant = []
# # for i in all_keyword:
# #     document_keyword = list(set(i))
# #     relevant.append(len(list(set(i))))
# # #print(relevant)
# #
# #
# # #print("retrieved")
# # retrieved = []
# # for i in retrieved_documents:
# #     retrieved.append(sum(i))
# # #print(retrieved)
# #
# #
# # Recall = []
# # for i in range(0, 1000):
# #     if relevant[i] == 0:
# #         Recall.append(0)
# #     else:
# #         Recall.append(float(sum(intersection[i]) / relevant[i]))
# #
# # #Recall_df = pd.DataFrame(data=Recall)
# # # print("Recall")
# # # print(str(Recall_df))
# # # print("Recall average:", max(Recall), sum(Recall)/1000.0)
# #
# #
# # Precision = []
# # for i in range(0, 1000):
# #     if retrieved[i] == 0:
# #         Precision.append(0)
# #     else:
# #         Precision.append(float(sum(intersection[i]) / retrieved[i]))
# #
# #
# # #Precision_df = pd.DataFrame(data=Precision)
# # # print("Precision")
# # # print(str(Precision_df))
# # # print("precision average:", max(Precision), sum(Precision)/1000.0)
# #
# # F_score = []
# # for i in range(0, 1000):
# #     if (Precision[i] + Recall[i]) == 0:
# #         F_score.append(0.0)
# #     else:
# #         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
# #
# # #print(F_score)
# # #F_score_df = pd.DataFrame(data=F_score)
# # #print("F_score")
# # #print(str(F_score_df))
# # F_aver_cbow_150 = (sum(F_score)/1000.0)
# # print("cbow_150:", F_aver_cbow_150)
#
#
#
# ##Skip-gram evaluation..............................................................................................................................
#
# ### Skip-gram30-words_Retrieved document data..................................
# line_df1 = []
# with open("sg_10_score.out", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line_list = line.replace(',', ' ').replace('\n', ' ').split()
#         line_df = []
#         for num_str in line_list:
#             num = float(num_str)
#             line_df.append(num)
#         line_df1.append(line_df)
#         # print('line_df1:', line_df1)
#
#     all_line = []
#     one_line = []
#     count = 0
#     for i in line_df1:
#         one_line = []
#         Min = min(i)
#         Max = max(i)
#         for j in i:
#             one_line.append((j - Min) / (Max - Min))
#         all_line.append(one_line)
#     # print('all_line')
#     # print(all_line)
# # retrieved_df = pd.DataFrame(data=line_df1)
# # print('retrieved document')
# # print(str(retrieved_df))
# #
#
# # # # relevant and retrivent data...............................
# count = 0
# score1 = []
# intersection = []
# retrieved_documents = []
# top_keyword = all_abstract_Words[0:10]
# for i in range(0, 1000):
#     score = []
#     retrieved_score = []
#
#     for j in range(0, len(all_line[i])):
#         fl_data = float(all_line[i][j])
#         if fl_data > 0.5 and top_keyword[j] in all_keyword[i]:
#             score.append(1)
#         else:
#             score.append(0)
#
#         if fl_data > 0.5:
#             retrieved_score.append(1)
#         else:
#             retrieved_score.append(0)
#     intersection.append(score)
#     retrieved_documents.append(retrieved_score)
# #intersection_df = pd.DataFrame(data=intersection)
# # print('intersection_df')
# # print(str(intersection_df))
#
#
# #print("relevant")
# relevant = []
# for i in all_keyword:
#     document_keyword = list(set(i))
#     relevant.append(len(list(set(i))))
# #print(relevant)
#
#
# #print("retrieved")
# retrieved = []
# for i in retrieved_documents:
#     retrieved.append(sum(i))
# #print(retrieved)
#
#
# Recall = []
# for i in range(0, 1000):
#     if relevant[i] == 0:
#         Recall.append(0)
#     else:
#         Recall.append(float(sum(intersection[i]) / relevant[i]))
#
# #Recall_df = pd.DataFrame(data=Recall)
# # print("Recall")
# # print(str(Recall_df))
# # print("Recall average:", max(Recall), sum(Recall)/1000.0)
#
#
# Precision = []
# for i in range(0, 1000):
#     if retrieved[i] == 0:
#         Precision.append(0)
#     else:
#         Precision.append(float(sum(intersection[i]) / retrieved[i]))
#
#
# #Precision_df = pd.DataFrame(data=Precision)
# # print("Precision")
# # print(str(Precision_df))
# # print("precision average:", max(Precision), sum(Precision)/1000.0)
#
# F_score = []
# for i in range(0, 1000):
#     if (Precision[i] + Recall[i]) == 0:
#         F_score.append(0.0)
#     else:
#         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
#
# #print(F_score)
# #F_score_df = pd.DataFrame(data=F_score)
# #F_score_df.to_excel('F_score.xlsx')
# #print("F_score")
# #print(str(F_score_df))
# F_aver_sg_10 = (max(F_score))
# print("sg_10:",F_aver_sg_10)
#
#
# ### Skip-gram30-words_Retrieved document data..................................
# line_df1 = []
# with open("sg_20_score.out", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line_list = line.replace(',', ' ').replace('\n', ' ').split()
#         line_df = []
#         for num_str in line_list:
#             num = float(num_str)
#             line_df.append(num)
#         line_df1.append(line_df)
#         # print('line_df1:', line_df1)
#
#     all_line = []
#     one_line = []
#     count = 0
#     for i in line_df1:
#         one_line = []
#         Min = min(i)
#         Max = max(i)
#         for j in i:
#             one_line.append((j - Min) / (Max - Min))
#         all_line.append(one_line)
#     # print('all_line')
#     # print(all_line)
# # retrieved_df = pd.DataFrame(data=line_df1)
# # print('retrieved document')
# # print(str(retrieved_df))
# #
#
# # # # relevant and retrivent data...............................
# count = 0
# score1 = []
# intersection = []
# retrieved_documents = []
# top_keyword = all_abstract_Words[0:20]
# for i in range(0, 1000):
#     score = []
#     retrieved_score = []
#
#     for j in range(0, len(all_line[i])):
#         fl_data = float(all_line[i][j])
#         if fl_data > 0.5 and top_keyword[j] in all_keyword[i]:
#             score.append(1)
#         else:
#             score.append(0)
#
#         if fl_data > 0.5:
#             retrieved_score.append(1)
#         else:
#             retrieved_score.append(0)
#     intersection.append(score)
#     retrieved_documents.append(retrieved_score)
# #intersection_df = pd.DataFrame(data=intersection)
# # print('intersection_df')
# # print(str(intersection_df))
#
#
# #print("relevant")
# relevant = []
# for i in all_keyword:
#     document_keyword = list(set(i))
#     relevant.append(len(list(set(i))))
# #print(relevant)
#
#
# #print("retrieved")
# retrieved = []
# for i in retrieved_documents:
#     retrieved.append(sum(i))
# #print(retrieved)
#
#
# Recall = []
# for i in range(0, 1000):
#     if relevant[i] == 0:
#         Recall.append(0)
#     else:
#         Recall.append(float(sum(intersection[i]) / relevant[i]))
#
# #Recall_df = pd.DataFrame(data=Recall)
# # print("Recall")
# # print(str(Recall_df))
# # print("Recall average:", max(Recall), sum(Recall)/1000.0)
#
#
# Precision = []
# for i in range(0, 1000):
#     if retrieved[i] == 0:
#         Precision.append(0)
#     else:
#         Precision.append(float(sum(intersection[i]) / retrieved[i]))
#
#
# #Precision_df = pd.DataFrame(data=Precision)
# # print("Precision")
# # print(str(Precision_df))
# # print("precision average:", max(Precision), sum(Precision)/1000.0)
#
# F_score = []
# for i in range(0, 1000):
#     if (Precision[i] + Recall[i]) == 0:
#         F_score.append(0.0)
#     else:
#         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
#
# #print(F_score)
# #F_score_df = pd.DataFrame(data=F_score)
# #F_score_df.to_excel('F_score.xlsx')
# #print("F_score")
# #print(str(F_score_df))
# F_aver_sg_20 = (max(F_score))
# print("sg_20:", F_aver_sg_20)
#
#
#
#
# ### Skip-gram30-words_Retrieved document data..................................
# line_df1 = []
# with open("sg_30_score.out", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line_list = line.replace(',', ' ').replace('\n', ' ').split()
#         line_df = []
#         for num_str in line_list:
#             num = float(num_str)
#             line_df.append(num)
#         line_df1.append(line_df)
#         # print('line_df1:', line_df1)
#
#     all_line = []
#     one_line = []
#     count = 0
#     for i in line_df1:
#         one_line = []
#         Min = min(i)
#         Max = max(i)
#         for j in i:
#             one_line.append((j - Min) / (Max - Min))
#         all_line.append(one_line)
#     # print('all_line')
#     # print(all_line)
# # retrieved_df = pd.DataFrame(data=line_df1)
# # print('retrieved document')
# # print(str(retrieved_df))
# #
#
# # # # relevant and retrivent data...............................
# count = 0
# score1 = []
# intersection = []
# retrieved_documents = []
# top_keyword = all_abstract_Words[0:30]
# for i in range(0, 1000):
#     score = []
#     retrieved_score = []
#
#     for j in range(0, len(all_line[i])):
#         fl_data = float(all_line[i][j])
#         if fl_data > 0.5 and top_keyword[j] in all_keyword[i]:
#             score.append(1)
#         else:
#             score.append(0)
#
#         if fl_data > 0.5:
#             retrieved_score.append(1)
#         else:
#             retrieved_score.append(0)
#     intersection.append(score)
#     retrieved_documents.append(retrieved_score)
# #intersection_df = pd.DataFrame(data=intersection)
# # print('intersection_df')
# # print(str(intersection_df))
#
#
# #print("relevant")
# relevant = []
# for i in all_keyword:
#     document_keyword = list(set(i))
#     relevant.append(len(list(set(i))))
# #print(relevant)
#
#
# #print("retrieved")
# retrieved = []
# for i in retrieved_documents:
#     retrieved.append(sum(i))
# #print(retrieved)
#
#
# Recall = []
# for i in range(0, 1000):
#     if relevant[i] == 0:
#         Recall.append(0)
#     else:
#         Recall.append(float(sum(intersection[i]) / relevant[i]))
#
# #Recall_df = pd.DataFrame(data=Recall)
# # print("Recall")
# # print(str(Recall_df))
# # print("Recall average:", max(Recall), sum(Recall)/1000.0)
#
#
# Precision = []
# for i in range(0, 1000):
#     if retrieved[i] == 0:
#         Precision.append(0)
#     else:
#         Precision.append(float(sum(intersection[i]) / retrieved[i]))
#
#
# #Precision_df = pd.DataFrame(data=Precision)
# # print("Precision")
# # print(str(Precision_df))
# # print("precision average:", max(Precision), sum(Precision)/1000.0)
#
# F_score = []
# for i in range(0, 1000):
#     if (Precision[i] + Recall[i]) == 0:
#         F_score.append(0.0)
#     else:
#         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
#
# #print(F_score)
# #F_score_df = pd.DataFrame(data=F_score)
# #F_score_df.to_excel('F_score.xlsx')
# #print("F_score")
# #print(str(F_score_df))
# F_aver_sg_30 = (max(F_score))
# print("sg_30:",F_aver_sg_30)
#
#
#
#
# ### sg_60-words_Retrieved document data....................................................................
# line_df1 = []
# with open("sg_40_score.out", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line_list = line.replace(',', ' ').replace('\n', ' ').split()
#         line_df = []
#         for num_str in line_list:
#             num = float(num_str)
#             line_df.append(num)
#         line_df1.append(line_df)
#         # print('line_df1:', line_df1)
#
#     all_line = []
#     one_line = []
#     count = 0
#     for i in line_df1:
#         one_line = []
#         Min = min(i)
#         Max = max(i)
#         for j in i:
#             one_line.append((j - Min) / (Max - Min))
#         all_line.append(one_line)
#     # print('all_line')
#     # print(all_line)
# # retrieved_df = pd.DataFrame(data=line_df1)
# # print('retrieved document')
# # print(str(retrieved_df))
# #
#
# # # # relevant and retrivent data...............................
# count = 0
# score1 = []
# intersection = []
# retrieved_documents = []
# top_keyword = all_abstract_Words[0:60]
# for i in range(0, 1000):
#     score = []
#     retrieved_score = []
#
#     for j in range(0, len(all_line[i])):
#         fl_data = float(all_line[i][j])
#         if fl_data > 0.5 and top_keyword[j] in all_keyword[i]:
#             score.append(1)
#         else:
#             score.append(0)
#
#         if fl_data > 0.5:
#             retrieved_score.append(1)
#         else:
#             retrieved_score.append(0)
#     intersection.append(score)
#     retrieved_documents.append(retrieved_score)
# #intersection_df = pd.DataFrame(data=intersection)
# # print('intersection_df')
# # print(str(intersection_df))
#
#
# #print("relevant")
# relevant = []
# for i in all_keyword:
#     document_keyword = list(set(i))
#     relevant.append(len(list(set(i))))
# #print(relevant)
#
#
# #print("retrieved")
# retrieved = []
# for i in retrieved_documents:
#     retrieved.append(sum(i))
# #print(retrieved)
#
#
# Recall = []
# for i in range(0, 1000):
#     if relevant[i] == 0:
#         Recall.append(0)
#     else:
#         Recall.append(float(sum(intersection[i]) / relevant[i]))
#
# Recall_df = pd.DataFrame(data=Recall)
# # print("Recall")
# # print(str(Recall_df))
# # print("Recall average:", max(Recall), sum(Recall)/1000.0)
#
#
# Precision = []
# for i in range(0, 1000):
#     if retrieved[i] == 0:
#         Precision.append(0)
#     else:
#         Precision.append(float(sum(intersection[i]) / retrieved[i]))
#
#
# Precision_df = pd.DataFrame(data=Precision)
# # print("Precision")
# # print(str(Precision_df))
# # print("precision average:", max(Precision), sum(Precision)/1000.0)
#
# F_score = []
# for i in range(0, 1000):
#     if (Precision[i] + Recall[i]) == 0:
#         F_score.append(0.0)
#     else:
#         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
#
# #print(F_score)
# #F_score_df = pd.DataFrame(data=F_score)
# #print("F_score")
# #print(str(F_score_df))
# F_aver_sg_40 = (sum(F_score)/1000.0)
# print("sg_40:", F_aver_sg_40)
#
#
#
#
# ### CBOW90-words_Retrieved document data....................................................................
# line_df1 = []
# with open("sg_50_score.out", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line_list = line.replace(',', ' ').replace('\n', ' ').split()
#         line_df = []
#         for num_str in line_list:
#             num = float(num_str)
#             line_df.append(num)
#         line_df1.append(line_df)
#         # print('line_df1:', line_df1)
#
#     all_line = []
#     one_line = []
#     count = 0
#     for i in line_df1:
#         one_line = []
#         Min = min(i)
#         Max = max(i)
#         for j in i:
#             one_line.append((j - Min) / (Max - Min))
#         all_line.append(one_line)
#     # print('all_line')
#     # print(all_line)
# # retrieved_df = pd.DataFrame(data=line_df1)
# # print('retrieved document')
# # print(str(retrieved_df))
# #
#
# # # # relevant and retrivent data...............................
# count = 0
# score1 = []
# intersection = []
# retrieved_documents = []
# top_keyword = all_abstract_Words[0:90]
# for i in range(0, 1000):
#     score = []
#     retrieved_score = []
#
#     for j in range(0, len(all_line[i])):
#         fl_data = float(all_line[i][j])
#         if fl_data > 0.5 and top_keyword[j] in all_keyword[i]:
#             score.append(1)
#         else:
#             score.append(0)
#
#         if fl_data > 0.5:
#             retrieved_score.append(1)
#         else:
#             retrieved_score.append(0)
#     intersection.append(score)
#     retrieved_documents.append(retrieved_score)
# #intersection_df = pd.DataFrame(data=intersection)
# # print('intersection_df')
# # print(str(intersection_df))
#
#
# #print("relevant")
# relevant = []
# for i in all_keyword:
#     document_keyword = list(set(i))
#     relevant.append(len(list(set(i))))
# #print(relevant)
#
#
# #print("retrieved")
# retrieved = []
# for i in retrieved_documents:
#     retrieved.append(sum(i))
# #print(retrieved)
#
#
# Recall = []
# for i in range(0, 1000):
#     if relevant[i] == 0:
#         Recall.append(0)
#     else:
#         Recall.append(float(sum(intersection[i]) / relevant[i]))
#
# #Recall_df = pd.DataFrame(data=Recall)
# # print("Recall")
# # print(str(Recall_df))
# # print("Recall average:", max(Recall), sum(Recall)/1000.0)
#
#
# Precision = []
# for i in range(0, 1000):
#     if retrieved[i] == 0:
#         Precision.append(0)
#     else:
#         Precision.append(float(sum(intersection[i]) / retrieved[i]))
#
#
# #Precision_df = pd.DataFrame(data=Precision)
# #Precision_df.to_excel('Precision.xlsx')
# # print("Precision")
# # print(str(Precision_df))
# # print("precision average:", max(Precision), sum(Precision)/1000.0)
#
# F_score = []
# for i in range(0, 1000):
#     if (Precision[i] + Recall[i]) == 0:
#         F_score.append(0.0)
#     else:
#         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
#
# #print(F_score)
# #F_score_df = pd.DataFrame(data=F_score)
# #print("F_score")
# #print(str(F_score_df))
# F_aver_sg_50 = (sum(F_score)/1000.0)
# print("sg_50:", F_aver_sg_50)
#
#
#
#
# ### sg120-words_Retrieved document data....................................................................
# line_df1 = []
# with open("sg_120_score.out", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line_list = line.replace(',', ' ').replace('\n', ' ').split()
#         line_df = []
#         for num_str in line_list:
#             num = float(num_str)
#             line_df.append(num)
#         line_df1.append(line_df)
#         # print('line_df1:', line_df1)
#
#     all_line = []
#     one_line = []
#     count = 0
#     for i in line_df1:
#         one_line = []
#         Min = min(i)
#         Max = max(i)
#         for j in i:
#             one_line.append((j - Min) / (Max - Min))
#         all_line.append(one_line)
#     # print('all_line')
#     # print(all_line)
# # retrieved_df = pd.DataFrame(data=line_df1)
# # print('retrieved document')
# # print(str(retrieved_df))
# #
#
# # # # relevant and retrivent data...............................
# count = 0
# score1 = []
# intersection = []
# retrieved_documents = []
# top_keyword = all_abstract_Words[0:120]
# for i in range(0, 1000):
#     score = []
#     retrieved_score = []
#
#     for j in range(0, len(all_line[i])):
#         fl_data = float(all_line[i][j])
#         if fl_data > 0.1 and top_keyword[j] in all_keyword[i]:
#             score.append(1)
#         else:
#             score.append(0)
#
#         if fl_data > 0.1:
#             retrieved_score.append(1)
#         else:
#             retrieved_score.append(0)
#     intersection.append(score)
#     retrieved_documents.append(retrieved_score)
# #intersection_df = pd.DataFrame(data=intersection)
# # print('intersection_df')
# # print(str(intersection_df))
#
#
# #print("relevant")
# relevant = []
# for i in all_keyword:
#     document_keyword = list(set(i))
#     relevant.append(len(list(set(i))))
# #print(relevant)
#
#
# #print("retrieved")
# retrieved = []
# for i in retrieved_documents:
#     retrieved.append(sum(i))
# #print(retrieved)
#
#
# Recall = []
# for i in range(0, 1000):
#     if relevant[i] == 0:
#         Recall.append(0)
#     else:
#         Recall.append(float(sum(intersection[i]) / relevant[i]))
#
# #Recall_df = pd.DataFrame(data=Recall)
# # print("Recall")
# # print(str(Recall_df))
# # print("Recall average:", max(Recall), sum(Recall)/1000.0)
#
#
# Precision = []
# for i in range(0, 1000):
#     if retrieved[i] == 0:
#         Precision.append(0)
#     else:
#         Precision.append(float(sum(intersection[i]) / retrieved[i]))
#
#
# #Precision_df = pd.DataFrame(data=Precision)
# # print("Precision")
# # print(str(Precision_df))
# # print("precision average:", max(Precision), sum(Precision)/1000.0)
#
# F_score = []
# for i in range(0, 1000):
#     if (Precision[i] + Recall[i]) == 0:
#         F_score.append(0.0)
#     else:
#         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
#
# #print(F_score)
# #F_score_df = pd.DataFrame(data=F_score)
# #F_score_df.to_excel('F_score.xlsx')
# #print("F_score")
# #print(str(F_score_df))
# F_aver_sg_120 = (sum(F_score)/1000.0)
# print("sg_120:",F_aver_sg_120)
#
#
#
#
# ### sg150-words_Retrieved document data....................................................................
# line_df1 = []
# with open("sg_150_score.out", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line_list = line.replace(',', ' ').replace('\n', ' ').split()
#         line_df = []
#         for num_str in line_list:
#             num = float(num_str)
#             line_df.append(num)
#         line_df1.append(line_df)
#         # print('line_df1:', line_df1)
#
#     all_line = []
#     one_line = []
#     count = 0
#     for i in line_df1:
#         one_line = []
#         Min = min(i)
#         Max = max(i)
#         for j in i:
#             one_line.append((j - Min) / (Max - Min))
#         all_line.append(one_line)
#     # print('all_line')
#     # print(all_line)
# # retrieved_df = pd.DataFrame(data=line_df1)
# # print('retrieved document')
# # print(str(retrieved_df))
# #
#
# # # # relevant and retrivent data...............................
# count = 0
# score1 = []
# intersection = []
# retrieved_documents = []
# top_keyword = all_abstract_Words[0:150]
# for i in range(0, 1000):
#     score = []
#     retrieved_score = []
#
#     for j in range(0, len(all_line[i])):
#         fl_data = float(all_line[i][j])
#         if fl_data > 0.1 and top_keyword[j] in all_keyword[i]:
#             score.append(1)
#         else:
#             score.append(0)
#
#         if fl_data > 0.1:
#             retrieved_score.append(1)
#         else:
#             retrieved_score.append(0)
#     intersection.append(score)
#     retrieved_documents.append(retrieved_score)
# #intersection_df = pd.DataFrame(data=intersection)
# # print('intersection_df')
# # print(str(intersection_df))
#
#
# #print("relevant")
# relevant = []
# for i in all_keyword:
#     document_keyword = list(set(i))
#     relevant.append(len(list(set(i))))
# #print(relevant)
#
#
# #print("retrieved")
# retrieved = []
# for i in retrieved_documents:
#     retrieved.append(sum(i))
# #print(retrieved)
#
#
# Recall = []
# for i in range(0, 1000):
#     if relevant[i] == 0:
#         Recall.append(0)
#     else:
#         Recall.append(float(sum(intersection[i]) / relevant[i]))
#
# #Recall_df = pd.DataFrame(data=Recall)
# # print("Recall")
# # print(str(Recall_df))
# # print("Recall average:", max(Recall), sum(Recall)/1000.0)
#
#
# Precision = []
# for i in range(0, 1000):
#     if retrieved[i] == 0:
#         Precision.append(0)
#     else:
#         Precision.append(float(sum(intersection[i]) / retrieved[i]))
#
#
# Precision_df = pd.DataFrame(data=Precision)
# # print("Precision")
# # print(str(Precision_df))
# # print("precision average:", max(Precision), sum(Precision)/1000.0)
#
# F_score = []
# for i in range(0, 1000):
#     if (Precision[i] + Recall[i]) == 0:
#         F_score.append(0.0)
#     else:
#         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
#
# #print(F_score)
# #F_score_df = pd.DataFrame(data=F_score)
# #print("F_score")
# #print(str(F_score_df))
# F_aver_sg_150 = (sum(F_score)/1000.0)
# print("sg_150:", F_aver_sg_150)
#
#
#
#
# ##Tf-idf evaluation..............................................................................................................................
#
#
# ## tf-idf30-words_Retrieved document data..................................
# line_df1 = []
# with open("tfidf_10_score.out", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line_list = line.replace(',', ' ').replace('\n', ' ').split()
#         line_df = []
#         line_1 = []
#         for num_str in line_list:
#             num = float(num_str)
#             if num > 0.1:
#                 line_df.append(num)
#             else:
#                 line_df.append(0)
#         line_df1.append(line_df)
#     #print("tf-idf_30:", line_df1)
#
#
#     Last_Line_df1 = []
#     Last_Line_1 = []
#     count = 0
#     for Line_1 in line_df1:
#         Last_Line_1 = []
#         Sum = sum(Line_1)
#         for one_thing in Line_1:
#             if one_thing == 0:
#                 Last_Line_1.append(0)
#             else:
#                 Last_Line_1.append(one_thing / Sum)
# #        count += 1
#         Last_Line_df1.append(Last_Line_1)
#
#     #print("Last_Line_df1")
#     #print(Last_Line_df1)
#
#     final_test = []
#     test_result = []
#     for i in Last_Line_df1:
#         final_test = []
#         for j in i:
#             if j > 0.5:
#                 final_test.append(j)
#             else:
#                 final_test.append(0)
#         test_result.append(final_test)
#
# #retrieved_df = pd.DataFrame(data=test_result)
# # print('retrieved document')
# # print(str(retrieved_df))
#
#
# # # # relevant and retrivent data...............................
# count = 0
# score1 = []
# intersection = []
# retrieved_documents = []
# top_keyword = all_abstract_Words[0:10]
# for i in range(0, 1000):
#     score = []
#     retrieved_score = []
#
#     for j in range(0, len(test_result[i])):
#         fl_data = float(test_result[i][j])
#         if fl_data > 0 and top_keyword[j] in all_keyword[i]:
#             score.append(1)
#         else:
#             score.append(0)
#
#         if fl_data > 0:
#             retrieved_score.append(1)
#         else:
#             retrieved_score.append(0)
#     intersection.append(score)
#     retrieved_documents.append(retrieved_score)
# # intersection_df = pd.DataFrame(data=intersection)
# # print('intersection_df')
# # print(str(intersection_df))
#
#
# #print("relevant")
# relevant = []
# for i in all_keyword:
#     document_keyword = list(set(i))
#     relevant.append(len(list(set(i))))
# #print(relevant)
#
#
# #print("retrieved")
# retrieved = []
# for i in retrieved_documents:
#     retrieved.append(sum(i))
# #print(retrieved)
#
#
# Recall = []
# for i in range(0, 1000):
#     if relevant[i] == 0:
#         Recall.append(0)
#     else:
#         Recall.append(float(sum(intersection[i]) / relevant[i]))
#
# #Recall_df = pd.DataFrame(data=Recall)
# # print("Recall")
# # print(str(Recall_df))
#
#
#
# Precision = []
# for i in range(0, 1000):
#     if retrieved[i] == 0:
#         Precision.append(0)
#     else:
#         Precision.append(float(sum(intersection[i]) / retrieved[i]))
#
# #Precision_df = pd.DataFrame(data=Precision)
# # print("Precision")
# # print(str(Precision_df))
#
#
# F_score = []
# for i in range(0, 1000):
#     if (Precision[i] + Recall[i]) == 0:
#         F_score.append(0.0)
#     else:
#         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
#
# #print(F_score)
# #F_score_df = pd.DataFrame(data=F_score)
# #print(str(F_score_df))
# F_aver_tfidf_10 = (sum(F_score)/1000.0)
# print("tfidf_10:", F_aver_tfidf_10)
#
#
# ## tf-idf30-words_Retrieved document data..................................
# line_df1 = []
# with open("tfidf_20_score.out", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line_list = line.replace(',', ' ').replace('\n', ' ').split()
#         line_df = []
#         line_1 = []
#         for num_str in line_list:
#             num = float(num_str)
#             if num > 0.1:
#                 line_df.append(num)
#             else:
#                 line_df.append(0)
#         line_df1.append(line_df)
#     #print("tf-idf_30:", line_df1)
#
#
#     Last_Line_df1 = []
#     Last_Line_1 = []
#     count = 0
#     for Line_1 in line_df1:
#         Last_Line_1 = []
#         Sum = sum(Line_1)
#         for one_thing in Line_1:
#             if one_thing == 0:
#                 Last_Line_1.append(0)
#             else:
#                 Last_Line_1.append(one_thing / Sum)
# #        count += 1
#         Last_Line_df1.append(Last_Line_1)
#
#     #print("Last_Line_df1")
#     #print(Last_Line_df1)
#
#     final_test = []
#     test_result = []
#     for i in Last_Line_df1:
#         final_test = []
#         for j in i:
#             if j > 0.5:
#                 final_test.append(j)
#             else:
#                 final_test.append(0)
#         test_result.append(final_test)
#
# #retrieved_df = pd.DataFrame(data=test_result)
# # print('retrieved document')
# # print(str(retrieved_df))
#
#
# # # # relevant and retrivent data...............................
# count = 0
# score1 = []
# intersection = []
# retrieved_documents = []
# top_keyword = all_abstract_Words[0:20]
# for i in range(0, 1000):
#     score = []
#     retrieved_score = []
#
#     for j in range(0, len(test_result[i])):
#         fl_data = float(test_result[i][j])
#         if fl_data > 0 and top_keyword[j] in all_keyword[i]:
#             score.append(1)
#         else:
#             score.append(0)
#
#         if fl_data > 0:
#             retrieved_score.append(1)
#         else:
#             retrieved_score.append(0)
#     intersection.append(score)
#     retrieved_documents.append(retrieved_score)
# # intersection_df = pd.DataFrame(data=intersection)
# # print('intersection_df')
# # print(str(intersection_df))
#
#
# #print("relevant")
# relevant = []
# for i in all_keyword:
#     document_keyword = list(set(i))
#     relevant.append(len(list(set(i))))
# #print(relevant)
#
#
# #print("retrieved")
# retrieved = []
# for i in retrieved_documents:
#     retrieved.append(sum(i))
# #print(retrieved)
#
#
# Recall = []
# for i in range(0, 1000):
#     if relevant[i] == 0:
#         Recall.append(0)
#     else:
#         Recall.append(float(sum(intersection[i]) / relevant[i]))
#
# #Recall_df = pd.DataFrame(data=Recall)
# # print("Recall")
# # print(str(Recall_df))
#
#
#
# Precision = []
# for i in range(0, 1000):
#     if retrieved[i] == 0:
#         Precision.append(0)
#     else:
#         Precision.append(float(sum(intersection[i]) / retrieved[i]))
#
# #Precision_df = pd.DataFrame(data=Precision)
# # print("Precision")
# # print(str(Precision_df))
#
#
# F_score = []
# for i in range(0, 1000):
#     if (Precision[i] + Recall[i]) == 0:
#         F_score.append(0.0)
#     else:
#         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
#
# #print(F_score)
# #F_score_df = pd.DataFrame(data=F_score)
# #print(str(F_score_df))
# F_aver_tfidf_20 = (sum(F_score)/1000.0)
# print("tfidf_20:", F_aver_tfidf_20)
#
#
#
# ## tf-idf30-words_Retrieved document data..................................
# line_df1 = []
# with open("tfidf_30_score.out", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line_list = line.replace(',', ' ').replace('\n', ' ').split()
#         line_df = []
#         line_1 = []
#         for num_str in line_list:
#             num = float(num_str)
#             if num > 0.1:
#                 line_df.append(num)
#             else:
#                 line_df.append(0)
#         line_df1.append(line_df)
#     #print("tf-idf_30:", line_df1)
#
#
#     Last_Line_df1 = []
#     Last_Line_1 = []
#     count = 0
#     for Line_1 in line_df1:
#         Last_Line_1 = []
#         Sum = sum(Line_1)
#         for one_thing in Line_1:
#             if one_thing == 0:
#                 Last_Line_1.append(0)
#             else:
#                 Last_Line_1.append(one_thing / Sum)
# #        count += 1
#         Last_Line_df1.append(Last_Line_1)
#
#     #print("Last_Line_df1")
#     #print(Last_Line_df1)
#
#     final_test = []
#     test_result = []
#     for i in Last_Line_df1:
#         final_test = []
#         for j in i:
#             if j > 0.5:
#                 final_test.append(j)
#             else:
#                 final_test.append(0)
#         test_result.append(final_test)
#
# #retrieved_df = pd.DataFrame(data=test_result)
# # print('retrieved document')
# # print(str(retrieved_df))
#
#
# # # # relevant and retrivent data...............................
# count = 0
# score1 = []
# intersection = []
# retrieved_documents = []
# top_keyword = all_abstract_Words[0:30]
# for i in range(0, 1000):
#     score = []
#     retrieved_score = []
#
#     for j in range(0, len(test_result[i])):
#         fl_data = float(test_result[i][j])
#         if fl_data > 0 and top_keyword[j] in all_keyword[i]:
#             score.append(1)
#         else:
#             score.append(0)
#
#         if fl_data > 0:
#             retrieved_score.append(1)
#         else:
#             retrieved_score.append(0)
#     intersection.append(score)
#     retrieved_documents.append(retrieved_score)
# # intersection_df = pd.DataFrame(data=intersection)
# # print('intersection_df')
# # print(str(intersection_df))
#
#
# #print("relevant")
# relevant = []
# for i in all_keyword:
#     document_keyword = list(set(i))
#     relevant.append(len(list(set(i))))
# #print(relevant)
#
#
# #print("retrieved")
# retrieved = []
# for i in retrieved_documents:
#     retrieved.append(sum(i))
# #print(retrieved)
#
#
# Recall = []
# for i in range(0, 1000):
#     if relevant[i] == 0:
#         Recall.append(0)
#     else:
#         Recall.append(float(sum(intersection[i]) / relevant[i]))
#
# #Recall_df = pd.DataFrame(data=Recall)
# # print("Recall")
# # print(str(Recall_df))
#
#
#
# Precision = []
# for i in range(0, 1000):
#     if retrieved[i] == 0:
#         Precision.append(0)
#     else:
#         Precision.append(float(sum(intersection[i]) / retrieved[i]))
#
# #Precision_df = pd.DataFrame(data=Precision)
# # print("Precision")
# # print(str(Precision_df))
#
#
# F_score = []
# for i in range(0, 1000):
#     if (Precision[i] + Recall[i]) == 0:
#         F_score.append(0.0)
#     else:
#         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
#
# #print(F_score)
# #F_score_df = pd.DataFrame(data=F_score)
# #print(str(F_score_df))
# F_aver_tfidf_30 = (sum(F_score)/1000.0)
# print("tfidf_30:", F_aver_tfidf_30)
#
#
#
#
# ### tf-idf60-words_Retrieved document data....................................................................
# line_df1 = []
# with open("tfidf_40_score.out", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line_list = line.replace(',', ' ').replace('\n', ' ').split()
#         line_df = []
#         line_1 = []
#         for num_str in line_list:
#             num = float(num_str)
#             if num > 0.1:
#                 line_df.append(num)
#             else:
#                 line_df.append(0)
#         line_df1.append(line_df)
#     #print("tf-idf_60:", line_df1)
#
#
#     Last_Line_df1 = []
#     Last_Line_1 = []
#     count = 0
#     for Line_1 in line_df1:
#         Last_Line_1 = []
#         Sum = sum(Line_1)
#         for one_thing in Line_1:
#             if one_thing == 0:
#                 Last_Line_1.append(0)
#             else:
#                 Last_Line_1.append(one_thing / Sum)
# #        count += 1
#         Last_Line_df1.append(Last_Line_1)
#
#     #print("Last_Line_df1")
#     #print(Last_Line_df1)
#
#     final_test = []
#     test_result = []
#     for i in Last_Line_df1:
#         final_test = []
#         for j in i:
#             if j > 0.5:
#                 final_test.append(j)
#             else:
#                 final_test.append(0)
#         test_result.append(final_test)
#
# #retrieved_df = pd.DataFrame(data=test_result)
# # print('retrieved document')
# # print(str(retrieved_df))
#
#
# # # # relevant and retrivent data...............................
# count = 0
# score1 = []
# intersection = []
# retrieved_documents = []
# top_keyword = all_abstract_Words[0:60]
# for i in range(0, 1000):
#     score = []
#     retrieved_score = []
#
#     for j in range(0, len(test_result[i])):
#         fl_data = float(test_result[i][j])
#         if fl_data > 0 and top_keyword[j] in all_keyword[i]:
#             score.append(1)
#         else:
#             score.append(0)
#
#         if fl_data > 0:
#             retrieved_score.append(1)
#         else:
#             retrieved_score.append(0)
#     intersection.append(score)
#     retrieved_documents.append(retrieved_score)
# # print('intersection_df')
# # print(str(intersection_df))
#
#
# #print("relevant")
# relevant = []
# for i in all_keyword:
#     document_keyword = list(set(i))
#     relevant.append(len(list(set(i))))
# #print(relevant)
#
#
# #print("retrieved")
# retrieved = []
# for i in retrieved_documents:
#     retrieved.append(sum(i))
# #print(retrieved)
#
#
# Recall = []
# for i in range(0, 1000):
#     if relevant[i] == 0:
#         Recall.append(0)
#     else:
#         Recall.append(float(sum(intersection[i]) / relevant[i]))
#
# #Recall_df = pd.DataFrame(data=Recall)
# # print("Recall")
# # print(str(Recall_df))
# # print("Recall average:", max(Recall), sum(Recall)/1000.0)
#
#
# Precision = []
# for i in range(0, 1000):
#     if retrieved[i] == 0:
#         Precision.append(0)
#     else:
#         Precision.append(float(sum(intersection[i]) / retrieved[i]))
#
#
# #Precision_df = pd.DataFrame(data=Precision)
# # print("Precision")
# # print(str(Precision_df))
# # print("precision average:", max(Precision), sum(Precision)/1000.0)
#
# F_score = []
# for i in range(0, 1000):
#     if (Precision[i] + Recall[i]) == 0:
#         F_score.append(0.0)
#     else:
#         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
#
# #print(F_score)
# #F_score_df = pd.DataFrame(data=F_score)
# #print("F_score")
# #print(str(F_score_df))
# F_aver_tfidf_40 = (sum(F_score)/1000.0)
# print("tfidf_40:", F_aver_tfidf_40)
#
#
#
#
# ### tf-idf90-words_Retrieved document data....................................................................
# line_df1 = []
# with open("tfidf_50_score.out", 'r') as f:
#     lines = f.readlines()
#     for line in lines:
#         line_list = line.replace(',', ' ').replace('\n', ' ').split()
#         line_df = []
#         line_1 = []
#         for num_str in line_list:
#             num = float(num_str)
#             if num > 0.1:
#                 line_df.append(num)
#             else:
#                 line_df.append(0)
#         line_df1.append(line_df)
#     #print("tf-idf_90:", line_df1)
#
#
#     Last_Line_df1 = []
#     Last_Line_1 = []
#     count = 0
#     for Line_1 in line_df1:
#         Last_Line_1 = []
#         Sum = sum(Line_1)
#         for one_thing in Line_1:
#             if one_thing == 0:
#                 Last_Line_1.append(0)
#             else:
#                 Last_Line_1.append(one_thing / Sum)
# #        count += 1
#         Last_Line_df1.append(Last_Line_1)
#
#     #print("Last_Line_df1")
#     #print(Last_Line_df1)
#
#     final_test = []
#     test_result = []
#     for i in Last_Line_df1:
#         final_test = []
#         for j in i:
#             if j > 0.5:
#                 final_test.append(j)
#             else:
#                 final_test.append(0)
#         test_result.append(final_test)
#
# #retrieved_df = pd.DataFrame(data=test_result)
# # retrieved_df.to_excel('retrievened_documents.xlsx')
# # print('retrieved document')
# # print(str(retrieved_df))
#
#
# # # # relevant and retrivent data...............................
# count = 0
# score1 = []
# intersection = []
# retrieved_documents = []
# top_keyword = all_abstract_Words[0:90]
# for i in range(0, 1000):
#     score = []
#     retrieved_score = []
#
#     for j in range(0, len(test_result[i])):
#         fl_data = float(test_result[i][j])
#         if fl_data > 0 and top_keyword[j] in all_keyword[i]:
#             score.append(1)
#         else:
#             score.append(0)
#
#         if fl_data > 0:
#             retrieved_score.append(1)
#         else:
#             retrieved_score.append(0)
#     intersection.append(score)
#     retrieved_documents.append(retrieved_score)
# # print(str(intersection_df))
#
#
# #print("relevant")
# relevant = []
# for i in all_keyword:
#     document_keyword = list(set(i))
#     relevant.append(len(list(set(i))))
# #print(relevant)
#
#
# #print("retrieved")
# retrieved = []
# for i in retrieved_documents:
#     retrieved.append(sum(i))
# #print(retrieved)
#
#
# Recall = []
# for i in range(0, 1000):
#     if relevant[i] == 0:
#         Recall.append(0)
#     else:
#         Recall.append(float(sum(intersection[i]) / relevant[i]))
#
# #Recall_df = pd.DataFrame(data=Recall)
# # print("Recall")
# # print(str(Recall_df))
# # print("Recall average:", max(Recall), sum(Recall)/1000.0)
#
#
# Precision = []
# for i in range(0, 1000):
#     if retrieved[i] == 0:
#         Precision.append(0)
#     else:
#         Precision.append(float(sum(intersection[i]) / retrieved[i]))
#
#
# #Precision_df = pd.DataFrame(data=Precision)
# # print("Precision")
# # print(str(Precision_df))
# # print("precision average:", max(Precision), sum(Precision)/1000.0)
#
# F_score = []
# for i in range(0, 1000):
#     if (Precision[i] + Recall[i]) == 0:
#         F_score.append(0.0)
#     else:
#         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
#
# #print(F_score)
# #F_score_df = pd.DataFrame(data=F_score)
# #print("F_score")
# #print(str(F_score_df))
# F_aver_tfidf_50 = (sum(F_score)/1000.0)
# print("tfidf_50:", F_aver_tfidf_50)
#
#
#
# #
# # ### tf-idf120-words_Retrieved document data....................................................................
# # line_df1 = []
# # with open("tfidf_120_score.out", 'r') as f:
# #     lines = f.readlines()
# #     for line in lines:
# #         line_list = line.replace(',', ' ').replace('\n', ' ').split()
# #         line_df = []
# #         line_1 = []
# #         for num_str in line_list:
# #             num = float(num_str)
# #             if num > 0.1:
# #                 line_df.append(num)
# #             else:
# #                 line_df.append(0)
# #         line_df1.append(line_df)
# #     #print("tf-idf_120:", line_df1)
# #
# #
# #     Last_Line_df1 = []
# #     Last_Line_1 = []
# #     count = 0
# #     for Line_1 in line_df1:
# #         Last_Line_1 = []
# #         Sum = sum(Line_1)
# #         for one_thing in Line_1:
# #             if one_thing == 0:
# #                 Last_Line_1.append(0)
# #             else:
# #                 Last_Line_1.append(one_thing / Sum)
# #         Last_Line_df1.append(Last_Line_1)
# #
# #     #print("Last_Line_df1")
# #     #print(Last_Line_df1)
# #
# #     final_test = []
# #     test_result = []
# #     for i in Last_Line_df1:
# #         final_test = []
# #         for j in i:
# #             if j > 0.1:
# #                 final_test.append(j)
# #             else:
# #                 final_test.append(0)
# #         test_result.append(final_test)
# #
# # #retrieved_df = pd.DataFrame(data=test_result)
# # # retrieved_df.to_excel('retrievened_documents.xlsx')
# # # print('retrieved document')
# # # print(str(retrieved_df))
# #
# #
# # # # # relevant and retrivent data...............................
# # count = 0
# # score1 = []
# # intersection = []
# # retrieved_documents = []
# # top_keyword = all_abstract_Words[0:120]
# # for i in range(0, 1000):
# #     score = []
# #     retrieved_score = []
# #
# #     for j in range(0, len(test_result[i])):
# #         fl_data = float(test_result[i][j])
# #         if fl_data > 0 and top_keyword[j] in all_keyword[i]:
# #             score.append(1)
# #         else:
# #             score.append(0)
# #
# #         if fl_data > 0:
# #             retrieved_score.append(1)
# #         else:
# #             retrieved_score.append(0)
# #     intersection.append(score)
# #     retrieved_documents.append(retrieved_score)
# # # print('intersection_df')
# # # print(str(intersection_df))
# #
# #
# # #print("relevant")
# # relevant = []
# # for i in all_keyword:
# #     document_keyword = list(set(i))
# #     relevant.append(len(list(set(i))))
# # #print(relevant)
# #
# #
# # #print("retrieved")
# # retrieved = []
# # for i in retrieved_documents:
# #     retrieved.append(sum(i))
# # #print(retrieved)
# #
# #
# # Recall = []
# # for i in range(0, 1000):
# #     if relevant[i] == 0:
# #         Recall.append(0)
# #     else:
# #         Recall.append(float(sum(intersection[i]) / relevant[i]))
# #
# # #Recall_df = pd.DataFrame(data=Recall)
# # # print("Recall")
# # # print(str(Recall_df))
# # # print("Recall average:", max(Recall), sum(Recall)/1000.0)
# #
# #
# # Precision = []
# # for i in range(0, 1000):
# #     if retrieved[i] == 0:
# #         Precision.append(0)
# #     else:
# #         Precision.append(float(sum(intersection[i]) / retrieved[i]))
# #
# #
# # #Precision_df = pd.DataFrame(data=Precision)
# # # print("Precision")
# # # print(str(Precision_df))
# # # print("precision average:", max(Precision), sum(Precision)/1000.0)
# #
# # F_score = []
# # for i in range(0, 1000):
# #     if (Precision[i] + Recall[i]) == 0:
# #         F_score.append(0.0)
# #     else:
# #         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
# #
# # #print(F_score)
# # #F_score_df = pd.DataFrame(data=F_score)
# # #print("F_score")
# # #print(str(F_score_df))
# # F_aver_tfidf_120 = (sum(F_score)/1000.0)
# # print("tfidf_120:",F_aver_tfidf_120)
# #
# #
# #
# #
# # ### tf_idf_150-words_Retrieved document data....................................................................
# # line_df1 = []
# # with open("tfidf_150_score.out", 'r') as f:
# #     lines = f.readlines()
# #     for line in lines:
# #         line_list = line.replace(',', ' ').replace('\n', ' ').split()
# #         line_df = []
# #         line_1 = []
# #         for num_str in line_list:
# #             num = float(num_str)
# #             if num > 0.1:
# #                 line_df.append(num)
# #             else:
# #                 line_df.append(0)
# #         line_df1.append(line_df)
# #     #print("tf-idf_150:", line_df1)
# #
# #
# #     Last_Line_df1 = []
# #     Last_Line_1 = []
# #     count = 0
# #     for Line_1 in line_df1:
# #         Last_Line_1 = []
# #         Sum = sum(Line_1)
# #         for one_thing in Line_1:
# #             if one_thing == 0:
# #                 Last_Line_1.append(0)
# #             else:
# #                 Last_Line_1.append(one_thing / Sum)
# # #        count += 1
# #         Last_Line_df1.append(Last_Line_1)
# #
# #     #print("Last_Line_df1")
# #     #print(Last_Line_df1)
# #
# #     final_test = []
# #     test_result = []
# #     for i in Last_Line_df1:
# #         final_test = []
# #         for j in i:
# #             if j > 0.1:
# #                 final_test.append(j)
# #             else:
# #                 final_test.append(0)
# #         test_result.append(final_test)
# #
# # #retrieved_df = pd.DataFrame(data=test_result)
# # # retrieved_df.to_excel('retrievened_documents.xlsx')
# # # print('retrieved document')
# # # print(str(retrieved_df))
# #
# #
# # # # # relevant and retrivent data...............................
# # count = 0
# # score1 = []
# # intersection = []
# # retrieved_documents = []
# # top_keyword = all_abstract_Words[0:150]
# # for i in range(0, 1000):
# #     score = []
# #     retrieved_score = []
# #
# #     for j in range(0, len(test_result[i])):
# #         fl_data = float(test_result[i][j])
# #         if fl_data > 0 and top_keyword[j] in all_keyword[i]:
# #             score.append(1)
# #         else:
# #             score.append(0)
# #
# #         if fl_data > 0:
# #             retrieved_score.append(1)
# #         else:
# #             retrieved_score.append(0)
# #     intersection.append(score)
# #     retrieved_documents.append(retrieved_score)
# # #intersection_df = pd.DataFrame(data=intersection)
# # # print('intersection_df')
# # # print(str(intersection_df))
# #
# #
# # #print("relevant")
# # relevant = []
# # for i in all_keyword:
# #     document_keyword = list(set(i))
# #     relevant.append(len(list(set(i))))
# # #print(relevant)
# #
# #
# # #print("retrieved")
# # retrieved = []
# # for i in retrieved_documents:
# #     retrieved.append(sum(i))
# # #print(retrieved)
# #
# #
# # Recall = []
# # for i in range(0, 1000):
# #     if relevant[i] == 0:
# #         Recall.append(0)
# #     else:
# #         Recall.append(float(sum(intersection[i]) / relevant[i]))
# #
# # #Recall_df = pd.DataFrame(data=Recall)
# # # print("Recall")
# # # print(str(Recall_df))
# #
# #
# #
# # Precision = []
# # for i in range(0, 1000):
# #     if retrieved[i] == 0:
# #         Precision.append(0)
# #     else:
# #         Precision.append(float(sum(intersection[i]) / retrieved[i]))
# #
# #
# # #Precision_df = pd.DataFrame(data=Precision)
# # # print("Precision")
# # # print(str(Precision_df))
# # # print("precision average:", max(Precision), sum(Precision)/1000.0)
# #
# # F_score = []
# # for i in range(0, 1000):
# #     if (Precision[i] + Recall[i]) == 0:
# #         F_score.append(0.0)
# #     else:
# #         F_score.append(2.0 * (Precision[i] * Recall[i]) / (Precision[i] + Recall[i]))
# #
# # #print(F_score)
# # #F_score_df = pd.DataFrame(data=F_score)
# # #print(str(F_score_df))
# # #F_aver_tfidf_150 = (max(F_score), sum(F_score)/1000.0)
# # F_aver_tfidf_150 = (sum(F_score)/1000.0)
# # print("tfidf_150:", F_aver_tfidf_150)
#
#
#
# ##Evaluation Graph...................................................................
# #set width of bars
# barWidth = 0.30
#
#
# cbow = [F_aver_cbow_10, F_aver_cbow_20, F_aver_cbow_30, F_aver_cbow_40, F_aver_cbow_50]
# sg = [F_aver_sg_10, F_aver_sg_20, F_aver_sg_30, F_aver_sg_40, F_aver_sg_50]
# tf_idf = [F_aver_tfidf_10, F_aver_tfidf_20, F_aver_tfidf_30, F_aver_tfidf_40, F_aver_tfidf_50]
#
# # Set position of bar on X axis
# r1 = np.arange(len(cbow))
# r2 = [x + barWidth for x in r1]
# r3 = [x + barWidth for x in r2]
#
# # Make the plot
# plt.bar(r1, cbow, color='#a3acff', width=barWidth, edgecolor='white', label='CBOW') #
# plt.bar(r2, sg, color='#f9bc86', width=barWidth, edgecolor='white', label='Skip-gram', hatch="//") #
# plt.bar(r3, tf_idf, color='#73C2FB', width=barWidth, edgecolor='white', label='TF-IDF', hatch="x") #
#
#
# #Add xticks on the middle of the group bars
# plt.xlabel('Top-N', fontweight='bold')
# plt.ylabel('F-Scores', fontweight='bold')
# #plt.xticks(['10_words', '20_words', '30_words', '40_words', '50_words']) #[r + barWidth for r in range(len(cbow))],
# plt.ylim(0, 30)
# plt.grid()
#
# # Create legend & Show graphic
# plt.legend()
# plt.show()
#


#################################################################################################
# def autolable(rectangle_group):
#     for rect in rectangle_group:
#         height = rect.get_height()
#
#         ax.annotate(str(height),
#                     xy=(rect.get_x() + rect.get_width()*0.5, height),
#                     ha='center', xytext=(0, 3), textcoords='offset points', rotation=90, va='bottom')
#
# cbow = [0.054, 0.062, 0.058, 0.059, 0.054]
# sg = [0.8, 0.6, 0.53, 0.065, 0.061]
# tf_idf = [0.033, 0.030, 0.023, 0.014, 0.011]
#
# xticks = ['10_words', '20_words', '30_words', '40_words', '50_words']
#
#
# width = 0.30
# x_cbow = [x - width for x in range(len(cbow))]
# x_sg =   [x for x in range(len(sg))]
# x_tf_idf = [x + width for x in range(len(tf_idf))]
#
# fig, ax = plt.subplots()
# ax1 = ax.twinx()
#
# rec1 = ax.bar(x_cbow, cbow, width, label = 'CBOW', color='#a3acff')
# rec2 = ax.bar(x_sg, sg, width, label= 'Sg', color='#f9bc86', hatch="//")
# rec3 = ax.bar(x_tf_idf, tf_idf, width, label='tf-idf', color='#73C2FB', hatch="x")
#
#
#
# ax.set_xticks(range(len(xticks)))
# ax.set_xticklabels(xticks, fontsize=10)
# ax.set_xlabel('Data Type', fontweight='bold')
# ax.set_ylabel('F-score', fontweight='bold')
# ax.set_ylim(0, 1.0)
# ax.legend()
#
#
# autolable(rec1)
# autolable(rec2)
# autolable(rec3)
#
# #plt.plot(cbow, color='g',linestyle='--', marker='o')
# plt.plot(sg, color='b',linestyle='--', marker='o')
# #plt.plot(tf_idf, color='r',linestyle='--', marker='o')
#
# plt.show()





## Evaluation Graph.....................................................................................................
#
# width = 1
# groupgap=1
# cbow = [F_aver_cbow_30, F_aver_cbow_60, F_aver_cbow_90, F_aver_cbow_120, F_aver_cbow_150]
# sg = [F_aver_sg_30, F_aver_sg_60, F_aver_sg_90, F_aver_sg_120, F_aver_sg_150]
# tf_idf = [F_aver_tfidf_30, F_aver_tfidf_60, F_aver_tfidf_90, F_aver_tfidf_120, F_aver_tfidf_150]
# x1 = np.arange(len(cbow))
# x2 = np.arange(len(sg))+groupgap+len(cbow)
# x3 = np.arange(len(tf_idf))+groupgap+len(sg)+groupgap+len(cbow)
# ind = np.concatenate((x1,x2,x3))
# fig, ax = plt.subplots()
# rects1 = ax.bar(x1, cbow, width, color='r',  edgecolor= "white", label="cbow")
# rects2 = ax.bar(x2, sg, width, color='#6593F5',  edgecolor= "white",  hatch="//", label="sg")
# rects3 = ax.bar(x3, tf_idf, width, color='#73C2FB',  edgecolor= "white",  hatch="x", label="tf-idf")
# ax.set_ylabel('F_scores', fontsize=14)
# ax.set_xticks(ind)
# #ax.set_xticklabels([r + width for r in range(len(ind))], ['30_words', '60_words', '90_words', '120_words', '150_words'])
# plt.xlabel('Data Type', fontsize=14)
# plt.legend(['cbow', 'sg', 'tf-idf'], loc='upper right')
# plt.show()

