import tensorflow as tf
import utils

def main():
    input_text = """'xssmall' has 2 GB 'ram', 2 'vcpu', costs 2.3
'xsmall' has 4 GB 'ram', 2 'vcpu', costs 3.3
'xmedium' has 8 GB 'ram', 4 'vcpu', costs 5.5
'xlarge' has 32 GB 'ram', 16 'vcpu', costs 10
'xxlarge' has 64 GB 'ram', 32 'vcpu', costs 25
at least 8 GB 'ram', 4 'vcpu'
select 1 type"""

    loaded_vec_model = tf.keras.models.load_model("models/text_vectorizer.keras")
    vectorizer = loaded_vec_model.layers[0]

    model = tf.keras.models.load_model('models/nlp_to_math_demo.keras')

    print(f"\nSample input: {input_text}")
    lp_vars = utils.process_lp_vars(input_text, vectorizer, model)
    output = utils.parse_dynamic_resources(lp_vars)
    print(output)

    print("\nsolve it using any lp solver.")
    print("Example: GLPK (GNU Linear Programming Kit)")
    print("$ glpsol --lp problem.lp -o result\n")


if __name__ == "__main__":
    main()
