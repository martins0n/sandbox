create () {
    cd etna && python get_data.py && cd ..
    python transform/raw_transform.py
    python transform/raw_to_etna_format.py
}

clean () {
    ls
}

"$@"
