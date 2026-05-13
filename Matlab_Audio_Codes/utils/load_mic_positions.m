function mic_pos = load_mic_positions(xml_file)
% LOAD_MIC_POSITIONS  Parse microphone Cartesian coordinates from an XML layout file.
%
% xml_file  Path to a file containing <pos x="..." y="..." z="..."/> elements.
%
% Returns mic_pos as [numMics x 3] with columns [x, y, z] in the same units as the file.

    try
        xml_data = xmlread(xml_file);
        pos_nodes = xml_data.getElementsByTagName('pos');

        num_mics = pos_nodes.getLength();
        mic_pos = zeros(num_mics, 3);

        for i = 0:num_mics - 1
            node = pos_nodes.item(i);
            mic_pos(i + 1, 1) = str2double(node.getAttribute('x'));
            mic_pos(i + 1, 2) = str2double(node.getAttribute('y'));
            mic_pos(i + 1, 3) = str2double(node.getAttribute('z'));
        end

        fprintf('Loaded %d microphone positions from %s\n', num_mics, xml_file);
    catch ME
        error('Failed to load mic positions: %s', ME.message);
    end
end
