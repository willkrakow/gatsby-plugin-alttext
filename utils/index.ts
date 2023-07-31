import sharp from "sharp";
import crypto from "crypto";
import type { Node } from "gatsby";

const isImageSharp = (node: Node) => {
    return node.internal.type && node.internal.type === 'ImageSharp';
}

const getImageBuffer = (path: string) => {
    return sharp(path).toBuffer();
}

const typeDefs = `
    """
    ML Image
    """
    type MLImage implements Node @infer {
      objects: [MLObject]!
      label: MLLabel
    }

    """
    ML Object
    """
    type MLObject @infer {
      bbox: [Float]!
      score: Float
      class: String
    }

    """
    ML Label
    """
    type MLLabel implements Node @infer {
      className: String
      probability: Float
    }
`;

interface CreateMLImageFields {
    id: string;
    altText: {
        objects: any[];
        label: any;
    };
    parent: any;
}

const createMlImageNodeFields = ({ id, altText, parent }: CreateMLImageFields) => ({
    // Data for the node.
    objects: altText.objects || [],
    label: altText.label || {},
    // Required fields.
    id,
    parent,
    children: [],
    internal: {
        type: `MLImage`,
        contentDigest: crypto
            .createHash(`md5`)
            .update(JSON.stringify(altText))
            .digest(`hex`),
    }
});



type PluginOptions = {
    images: string;
}

type PluginOptionsSchema = {
    [Property in keyof PluginOptions]: {
        type: string;
        required: boolean;
    }
}

const optionsSchema: PluginOptionsSchema = {
    images: {
        type: "string",
        required: true,
    }
}


const validatePluginOptions = (pluginOptions: any) => {
    const errors: string[] = [];
    Object.entries(optionsSchema).forEach(([key, schema]) => {
        if (schema.required && !pluginOptions[key as keyof PluginOptions]){
            errors.push(`"${key}" is required`)
        }

        if (schema.type !== typeof pluginOptions[key as keyof PluginOptions]){
            errors.push(`"${key} must be of type ${schema.type}`)
        }
    });

    return errors;
}


export default {isImageSharp, typeDefs, getImageBuffer, createMlImageNodeFields, validatePluginOptions}
