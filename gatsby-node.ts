import utils from './utils/index';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as coco from '@tensorflow-models/coco-ssd';
import * as tf from '@tensorflow/tfjs-node';
import type { GatsbyNode } from 'gatsby';

const { createMlImageNodeFields, isImageSharp, typeDefs, getImageBuffer, validatePluginOptions } = utils;

let model: mobilenet.MobileNet | null;
let objectModel: coco.ObjectDetection;
let count = 0;

const loadModels = async () => {
    if (!model) {
        model = await mobilenet.load();
    }

    if (!objectModel) {
        objectModel = await coco.load({
            base: "mobilenet_v2",
        });
    }
}

const destroyModels = async () => {
    objectModel.dispose()
    model = null;
}


const classifyImage = async (imageBuffer: Buffer, imageType: string) => {
    const image = tf.node.decodeImage(imageBuffer, 3);
    let imageForObjects: tf.Tensor3D | tf.Tensor4D | ImageData;
    switch (imageType) {
        case "image/png":
            imageForObjects = tf.node.decodePng(imageBuffer, 3);
            break;
        case "image/jpg":
        case "image/jpeg":
            imageForObjects = tf.node.decodeJpeg(imageBuffer, 3);
            break;
        default:
            imageForObjects = tf.node.decodeImage(imageBuffer)
            break;
    }
    const objects = await objectModel.detect(imageForObjects as tf.Tensor3D);
    const label = await model?.classify(image as tf.Tensor<tf.Rank.R3>);

    return {
        objects,
        label,
    }
}

export const onPreBootstrap: GatsbyNode['onPreBootstrap'] = async () => loadModels();

export const onPreInit: GatsbyNode['onPreInit'] = ({ reporter }, pluginOptions) => {
    reporter.info(`Classifying images to generate alt text`)
    const errors = validatePluginOptions(pluginOptions);
    if (errors.length > 0) {
        errors.forEach(err => reporter.error(err))
    }
}

export const createSchemaCustomization: GatsbyNode['createSchemaCustomization'] = async ({ actions }) => {
    actions.createTypes(typeDefs)
}

export const onPostBootstrap: GatsbyNode['onPostBootstrap'] = async ({
    actions, getNode, reporter, createNodeId, getNodesByType,
}, pluginOptions) => {

    const images = getNodesByType(`ImageSharp`);
    const { createNode } = actions;
    await Promise.all(images.map(async (node) => {
        if (!isImageSharp(node)) return;
        const parentNode = getNode(node.parent as string) as any;

        if (!parentNode.absolutePath?.includes(pluginOptions.images)) {
            return;
        }

        try {
            const imageBuffer = await getImageBuffer(parentNode.absolutePath);
            const altText = await classifyImage(imageBuffer, parentNode.internal.mediaType);

            const mlImageNodeFields = createMlImageNodeFields({
                id: createNodeId(parentNode.absolutePath),
                altText,
                parent: node.id,
            });

            count += 1;
            createNode(mlImageNodeFields);
        } catch (err) {
            reporter.error(err as Error)
            reporter.error(`Error classifying ${parentNode.base}`)
        }
    }));


}

export const onPostBuild: GatsbyNode['onPostBuild'] = async ({ reporter }) => {
    await destroyModels();

    reporter.info(`${count} ML Image nodes created`)
}