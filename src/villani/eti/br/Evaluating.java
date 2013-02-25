package villani.eti.br;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.TreeMap;

import mulan.classifier.MultiLabelLearnerBase;
import mulan.classifier.lazy.BRkNN;
import mulan.classifier.lazy.MLkNN;
import mulan.classifier.meta.HMC;
import mulan.classifier.transformation.ClassifierChain;
import mulan.classifier.transformation.LabelPowerset;
import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import mulan.evaluation.Evaluation;
import mulan.evaluation.Evaluator;
import mulan.evaluation.measure.AveragePrecision;
import mulan.evaluation.measure.Coverage;
import mulan.evaluation.measure.ErrorSetSize;
import mulan.evaluation.measure.ExampleBasedAccuracy;
import mulan.evaluation.measure.ExampleBasedFMeasure;
import mulan.evaluation.measure.ExampleBasedPrecision;
import mulan.evaluation.measure.ExampleBasedRecall;
import mulan.evaluation.measure.ExampleBasedSpecificity;
import mulan.evaluation.measure.HammingLoss;
import mulan.evaluation.measure.IsError;
import mulan.evaluation.measure.Measure;
import mulan.evaluation.measure.MicroFMeasure;
import mulan.evaluation.measure.MicroPrecision;
import mulan.evaluation.measure.MicroRecall;
import mulan.evaluation.measure.OneError;
import mulan.evaluation.measure.RankingLoss;
import mulan.evaluation.measure.SubsetAccuracy;
import weka.classifiers.lazy.IBk;

public class Evaluating {

	public static void run(String id, LogBuilder log,
			TreeMap<String, String> entradas) {

		boolean ehd = Boolean.parseBoolean(entradas.get("ehd"));
		boolean lbp = Boolean.parseBoolean(entradas.get("lbp"));
		boolean sift = Boolean.parseBoolean(entradas.get("sift"));
		boolean gabor = Boolean.parseBoolean(entradas.get("gabor"));
		boolean mlknn = Boolean.parseBoolean(entradas.get("mlknn"));
		boolean brknn = Boolean.parseBoolean(entradas.get("brknn"));
		boolean chain = Boolean.parseBoolean(entradas.get("chain"));
		boolean lp = Boolean.parseBoolean(entradas.get("lp"));
		boolean hmc = Boolean.parseBoolean(entradas.get("hmc"));
		String rotulos = entradas.get("rotulos");
		String[] tecnicas = { "Ehd", "Lbp", "Sift", "Gabor" };
		String[] classificadores = {"MLkNN", "BRkNN", "Chain", "LP", "HMC"};

		for (String tecnica : tecnicas) {

			if (tecnica.equals("Ehd") && !ehd) continue;
			if (tecnica.equals("Lbp") && !lbp) continue;
			if (tecnica.equals("Sift") && !sift) continue;
			if (tecnica.equals("Gabor") && !gabor) continue;

			for (String classificador : classificadores) {

				if (classificador.equals("MLkNN") && !mlknn) continue;
				if (classificador.equals("BRkNN") && !brknn) continue;
				if (classificador.equals("Chain") && !chain) continue;
				if (classificador.equals("LP") && !lp) continue;
				if (classificador.equals("HMC") && !hmc) continue;

				for (int i = 0; i < 10; i++) {

					String base = tecnica + "-Sub" + i + ".arff";
					MultiLabelInstances trainingSet = null;
					try {
						log.write(" - Instanciando conjunto de treinamento multirrótulo de treino");
						trainingSet = new MultiLabelInstances(tecnica + "/" + base, rotulos);
					} catch (InvalidDataFormatException idfe) {
						log.write(" - Erro no formato de dados ao instanciar conjunto multirrótulos de treino " + 
								base + ": " + idfe.getMessage());
						System.exit(0);
					}

					log.write(" - Instanciando classificador " + classificador);
					MultiLabelLearnerBase mlLearner = null;
					if (classificador.equals("MLkNN")) mlLearner = new MLkNN(); // default k=10
					if (classificador.equals("BRkNN")) mlLearner = new BRkNN(); // default k=10
					if (classificador.equals("Chain")) {
						IBk kNN = new IBk(10);
						mlLearner = new ClassifierChain(kNN);
					}
					if (classificador.equals("LP")) {
						IBk kNN = new IBk(10);
						mlLearner = new LabelPowerset(kNN);
					}
					if (classificador.equals("HMC")) mlLearner = new HMC();

					try {
						log.write(" - Construindo modelo do " + mlLearner.getClass() + " a partir do conjunto de treinamento " + base);
						mlLearner.build(trainingSet);
					} catch (Exception e) {
						log.write(" - Falha ao construir o modelo do classificador: " + e.getMessage());
						System.exit(0);
					}

					log.write(" - Instanciando avaliador");
					Evaluator avaliador = new Evaluator();

					log.write(" - Instanciando lista de medidas");
					ArrayList<Measure> medidas = new ArrayList<Measure>();
					medidas.add(new HammingLoss());
					medidas.add(new SubsetAccuracy());
					medidas.add(new ExampleBasedPrecision());
					medidas.add(new ExampleBasedRecall());
					medidas.add(new ExampleBasedFMeasure());
					medidas.add(new ExampleBasedAccuracy());
					medidas.add(new ExampleBasedSpecificity());
					int numOfLabels = trainingSet.getNumLabels();
					medidas.add(new MicroPrecision(numOfLabels));
					medidas.add(new MicroRecall(numOfLabels));
					medidas.add(new MicroFMeasure(numOfLabels));
					medidas.add(new AveragePrecision());
					medidas.add(new Coverage());
					medidas.add(new OneError());
					medidas.add(new IsError());
					medidas.add(new ErrorSetSize());
					medidas.add(new RankingLoss());

					for (int j = 1; j < 10; j++) {
						
						if(j == i) continue;

						String baseTreino = tecnica + "-Sub" + j + ".arff";

						MultiLabelInstances testSet = null;
						
						try {
							log.write(" - Instanciando conjunto multirrótulo de teste");
							testSet = new MultiLabelInstances(tecnica + "/" + baseTreino, rotulos);
						} catch (InvalidDataFormatException idfe) {
							log.write(" - Erro no formato de dados ao instanciar conjunto multirrótulos de teste " + 
									baseTreino + ": " + idfe.getMessage());
							System.exit(0);
						}

						log.write(" - Avaliando o modelo gerado pelo classificador " + mlLearner.getClass());
						Evaluation avaliacao = null;
						try {
							avaliacao = avaliador.evaluate(mlLearner, testSet, medidas);
						} catch (IllegalArgumentException iae) {
							log.write(" - Argumentos utilizados inválidos: " + iae.getMessage());
							System.exit(0);
						} catch (Exception e) {
							log.write(" - Falha ao avaliar o modelo: " + e.getMessage());
							System.exit(0);
						}

						log.write(" - Salvando resultado da avaliação");
						File resultado = new File(id + classificador + "-" + tecnica + "-Sub" + j + ".csv");
						try {
							FileWriter escritor = new FileWriter(resultado);
							escritor.write(avaliacao.toString());
							escritor.close();
						} catch (IOException ioe) {
							log.write(" - Falha ao salvar resultado da avaliação: " + ioe.getMessage());
							System.exit(0);
						}

					}

				}

			}

		}

	}
}