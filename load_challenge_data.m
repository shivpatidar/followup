function data = load_challenge_data(file)
    f = fopen(file, 'rt');
    try
        l = fgetl(f);
        column_names = strsplit(l, '|');
        data = dlmread(file, '|', 1, 0);
    catch ex
        fclose(f);
        rethrow(ex);
    end
    fclose(f);

    % Ignore SepsisLabel column if present.
    if strcmp(column_names(end), 'SepsisLabel')
        column_names = column_names(1 : end-1);
        data = data(:, 1 : end-1);
    end
end