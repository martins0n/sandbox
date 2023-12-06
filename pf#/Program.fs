open System
open Microsoft.SemanticKernel
open Microsoft.SemanticKernel.Connectors.AI.OpenAI
open DotNetEnv

[<EntryPoint>]
let main argv =
    DotNetEnv.Env.Load()
    
    printfn "Hello from F#"

    let builder = KernelBuilder()

    builder.WithOpenAIChatCompletionService(
        "gpt-3.5-turbo",
        DotNetEnv.Env.GetString("OPENAI_KEY")
    ) |> ignore

    // Alternative using OpenAI
    //builder.WithOpenAIChatCompletionService(
    //         "gpt-3.5-turbo",                  // OpenAI Model name
    //         "...your OpenAI API Key...")     // OpenAI API Key

    let kernel = builder.Build()

    let prompt = """{{$input}}

One line TLDR with the fewest words."""

    let openAIRequestSettings = OpenAIRequestSettings(MaxTokens = 100)

    let summarize _prompt settings_ = kernel.CreateSemanticFunction(_prompt, settings_ )

    let summarize = summarize prompt openAIRequestSettings

    let text1 = """
1st Law of Thermodynamics - Energy cannot be created or destroyed.
2nd Law of Thermodynamics - For a spontaneous process, the entropy of the universe increases.
3rd Law of Thermodynamics - A perfect crystal at zero Kelvin has zero entropy."""

    let text2 = """
1. An object at rest remains at rest, and an object in motion remains in motion at constant speed and in a straight line unless acted on by an unbalanced force.
2. The acceleration of an object depends on the mass of the object and the amount of force applied.
3. Whenever one object exerts a force on another object, the second object exerts an equal and opposite on the first."""

    let result1 = kernel.RunAsync(text1, summarize)
    let result2 = kernel.RunAsync(text2, summarize)



    let result1 = result1.Result.ToString()

    printfn "%s" result1
    
    0
