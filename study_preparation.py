import json
import copy
import pandas as pd


def create_texts_with_values_output(final_documents, parameter=['m_s_cos_r_cos', 'm_s_cos_r_sr', 'm_s_wn_r_cos', 'm_s_wn_r_sr']):
    texts_with_values = []
    for document in final_documents:
        text_with_values = copy.deepcopy(document["candidates"])
        offset = 0
        for i in range(len(text_with_values)):
            #print(text_with_values[i])
            if isinstance(text_with_values[i], list):
                for p in parameter:
                    text_with_values[i].append({
                        p: round(document[p][i-offset], 3)
                    })
            else:
                offset += 1
        texts_with_values += [text_with_values]
    return texts_with_values




def create_candidates_df(final_document):
    keys = [
        "candidates_only",
        "similarity_cossim",
        "similarity_wordnet",
        "relevance_cossim",
        "relevance_singlerank",
        'm_s_cos_r_cos', 'm_s_cos_r_sr', 'm_s_wn_r_cos', 'm_s_wn_r_sr'
    ]
    data = [final_document[key] for key in keys]
    df = pd.DataFrame(data)
    df_transposed = df.transpose()
    df_transposed.columns = keys
    df_transposed.insert(0, "index", list(range(len(df_transposed["candidates_only"]))))
    return df_transposed
    # df_transposed.sort_values("metric_scores_cossim")


def create_rank_columns(df, sort_by=['m_s_cos_r_cos', 'm_s_cos_r_sr', 'm_s_wn_r_cos', 'm_s_wn_r_sr']):
    sorted_df = df
    for column in sort_by:
        # df.sort_values("metric_scores_cossim")
        sorted_df = sorted_df.sort_values(by=column)
        sorted_df[str(column + "_rank")] = range(len(sorted_df))
    return sorted_df.sort_values("index")
    #dfa.sort_values(dfa.keys()[-1])




def compare_metric_scores(df):
    import scipy
    return scipy.stats.kendalltau(df['metric_scores_cossim_rank'], df['metric_scores_singlerank_rank'])

def create_and_save_html(text_with_values, df, html_doc_name="html_doc.html", additional_text=""):
    html_template_1 = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        /* Define styles for the tooltip */
        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
            color: blue;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 150px;
            background-color: #333;
            color: #fff;
            text-align: center;
            border-radius: 6px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            margin-left: -75px;
            opacity: 0;
            transition: opacity 0.2s;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }

        .text {
            margin: 50px;
            margin-top: 300px;
            font-size: x-large;
        }
        /* SIDEBAR */
        .sidebar {
            width: 250px;
            background-color: #333;
            color: #fff;
            position: fixed;
            top: 0;
            left: -250px;
            height: 100%;
            overflow-y: auto;
            transition: left 0.3s;
        }

        .sidebar.show {
            left: 0;
        }

        .sidebar ul {
            list-style-type: none;
            padding: 0;
        }

        .sidebar li {
            padding: 10px;
        }

        .sidebar li:hover {
            background-color: #555;
        }
    </style>
</head>
<body>
    <div class="sidebar">
        <ol>
        """
    #<li>Word 1 (Ranked)</li>
    #<li>Word 2 (Ranked)</li>
    html_template_2 = """
        </ol>
    </div>
    <p class="text"> 
    """
    html_template_3 = """
    </p>
    <script>
    // JavaScript to toggle the sidebar
    const sidebar = document.querySelector(".sidebar");
    const text = document.querySelector(".text");

    text.addEventListener("click", () => {
        sidebar.classList.toggle("show");
    });
    </script>
</body>
</html>
"""
    text = ""
    number_of_values = len([x for x in text_with_values[0] if isinstance(x, dict)])
    i_candidate = 0
    for element in text_with_values:
        if isinstance(element, list):
            text += str(' <span class="tooltip"> ' + ' '.join(element[0:-number_of_values]) + "(" + str(i_candidate) + ")"
                        + ' <span class="tooltiptext"> ' + repr(element[-number_of_values:]) + ' </span></span> ')
            i_candidate += 1
        else:
            text += str(" " + element + " ")
    ranking = df.sort_values("m_s_wn_r_cos_rank")["candidates_only"].iloc[::-1]
    ranking_string = ''.join(["<li> {word} ({position}) </li> \n".format(word=candidate, position=pos)
                              for candidate, pos in zip(ranking,ranking.index)])
    html_output = html_template_1 + ranking_string + html_template_2 + text + "<br><br>" \
                  + additional_text + html_template_3

    with open(html_doc_name, "w") as f:
        f.write(html_output)
    return ranking #, html_output


if __name__ == "__main__":
    from main import final_documents_path
    with open(final_documents_path, 'r') as file:
        documents = json.load(file)
    texts_wv = create_texts_with_values_output(documents)
    dfs = []
    for document, text_wv in zip(documents, texts_wv):
        df = create_rank_columns(create_candidates_df(document))
        dfs += [df]
        ranking = create_and_save_html(text_wv, df, html_doc_name="html/" + document["title"] + ".html")


"""
i = 0
for final_document in final_documents:
    df = create_candidates_df(final_document)
    create_and_save_html_hover_text(continuous_texts[i], str("html_doc " + final_document["title"]),
                                    # calculate similarity of the metric-approaches
                                    additional_text=str([(name, round(result, 5)) for name, result in zip(
                                        ["stat: ", "Pvalue: "],
                                        scipy.stats.kendalltau(df['metric_scores_cossim'],
                                                               df['metric_scores_singlerank']))
                                         ]))
    i += 1
"""
